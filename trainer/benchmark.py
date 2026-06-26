import copy
import glob
import json
import os
import re

import torch
from gliner2 import GLiNER2
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

try:
    from trainer.results_store import (
        compute_test_set_hash,
        derive_adapter_params,
        get_cached,
        load_store,
        put_result,
        register_test_set,
        save_store,
    )
except ImportError:
    from results_store import (
        compute_test_set_hash,
        derive_adapter_params,
        get_cached,
        load_store,
        put_result,
        register_test_set,
        save_store,
    )

DEFAULT_TEST_FOLDER = "data/test"
DEFAULT_LABELS = {
    "ticker": "A stock market ticker symbol, usually 1-5 letters, often preceded by a dollar sign (e.g., $AAPL, TSLA). MUST NOT be option strikes, prices, index names, or internet slang acronyms.",
    "company": "The name of a corporation, hedge fund, or business entity. MUST NOT be an uppercase ticker symbol, an index, or generic finance terms.",
}

console = Console()


def locate_adapter_weights(adapter_dir):
    """Pick which checkpoint subfolder of an adapter dir to evaluate.

    Prefers ``best/`` (the early-stopping optimal) when present, falling back
    to ``final/`` (the last-step save, all that exists for pre-early-stopping
    runs). Returns ``None`` if neither has a safetensors file.
    """
    for sub in ("best", "final"):
        candidate = os.path.join(adapter_dir, sub)
        if os.path.exists(os.path.join(candidate, "adapter_model.safetensors")):
            return candidate
    return None


def get_all_adapters(models_dir="./models"):
    """Scans the models directory and returns a sorted list of all valid adapters."""
    if not os.path.exists(models_dir):
        return []

    adapters = glob.glob(os.path.join(models_dir, "reddit_adapter*"))
    valid_adapters = []

    for adapter_dir in adapters:
        weights_path = locate_adapter_weights(adapter_dir)
        if weights_path is None:
            continue

        folder_name = os.path.basename(adapter_dir)
        match = re.search(r"_v(\d+)$", folder_name)
        if match:
            v = int(match.group(1))
        elif folder_name == "reddit_adapter":
            v = 1
        else:
            continue

        valid_adapters.append(
            {
                "version": v,
                "name": f"GLiNER2 Large + Adapter v{v}",
                "path": weights_path,
            }
        )

    valid_adapters.sort(key=lambda x: x["version"])
    return valid_adapters


def parse_all_label_studio_exports(folder_path):
    """Parses all Label Studio JSON exports in a folder into clean GLiNER format."""
    if not os.path.isdir(folder_path):
        console.print(f"[red]Error: Folder {folder_path} not found.[/red]")
        return []

    clean_dataset = []
    files = glob.glob(os.path.join(folder_path, "*.json"))

    if not files:
        console.print(f"[red]No JSON files found in {folder_path}.[/red]")
        return clean_dataset

    for export_path in files:
        with open(export_path, "r", encoding="utf-8") as f:
            ls_data = json.load(f)

        for task in ls_data:
            if not task.get("annotations") or task["annotations"][0].get(
                "was_cancelled"
            ):
                continue

            text = task["data"]["text"]
            entities = []

            results = task["annotations"][0].get("result", [])
            for r in results:
                if r.get("type") == "labels":
                    val = r["value"]
                    entities.append(
                        {
                            "start": val["start"],
                            "end": val["end"],
                            "label": val["labels"][0],
                        }
                    )

            clean_dataset.append({"text": text, "entities": entities})

    return clean_dataset


def normalize_entity(surface):
    """Canonicalize an entity surface form for set-based, dedup-per-document scoring.

    The engine's public API (``recognize`` / ``recognize_ai``) returns a
    *deduplicated set* of tickers per text — the project cares about *what* a
    post is talking about, not how many times a symbol is repeated. Scoring
    therefore treats an entity as caught when it appears at least once in a
    document, and collapses repeated mentions of the same symbol to one key.

    Folding ``$`` prefixes, surrounding whitespace and case makes ``$GME``,
    ``GME``, ``gme`` and ``GME `` all compare equal (and ``$ AUG`` → ``AUG``).
    """
    collapsed = re.sub(r"\s+", " ", surface).strip()
    return collapsed.lstrip("$").strip().upper()


def calculate_metrics(tp, fp, fn):
    """Helper to safely calculate P, R, F1"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return {"p": precision, "r": recall, "f1": f1}


def chunk_text_for_inference(text, chunk_word_size=150, overlap_words=40):
    """
    Slices text into overlapping chunks, returning the chunk string and its
    absolute character start index so predictions can be mapped back accurately.
    """
    matches = list(re.finditer(r"\S+", text))
    chunks = []
    if not matches:
        return [(text, 0)]

    step_size = max(1, chunk_word_size - overlap_words)

    for i in range(0, len(matches), step_size):
        chunk_matches = matches[i : i + chunk_word_size]
        if not chunk_matches:
            break

        start_char = chunk_matches[0].start()
        end_char = chunk_matches[-1].end()

        chunk_text = text[start_char:end_char]
        chunks.append((chunk_text, start_char))

        if i + chunk_word_size >= len(matches):
            break

    return chunks


def evaluate_model(
    model,
    flat_chunks,
    doc_chunk_ranges,
    gold_per_doc,
    gold_by_label_per_doc,
    model_name="Model",
    label_descriptions=None,
    batch_size=128,
    progress_context=None,
    threshold=0.75,
):
    """Calculates NER metrics by running batched inference across the whole dataset.

    Parameters
    ----------
    flat_chunks : list of (doc_idx, chunk_text, char_offset)
        All chunks across all documents, flattened. ``doc_idx`` indexes into
        ``gold_per_doc`` / ``gold_by_label_per_doc``.
    doc_chunk_ranges : list of (start, end)
        Half-open slices into ``flat_chunks`` for each document.
    """
    labels_to_pass = label_descriptions if label_descriptions else ["ticker", "company"]
    label_keys = (
        list(labels_to_pass.keys())
        if isinstance(labels_to_pass, dict)
        else labels_to_pass
    )

    metrics_counts = {k: {"tp": 0, "fp": 0, "fn": 0} for k in label_keys}
    metrics_counts["overall"] = {"tp": 0, "fp": 0, "fn": 0}

    progress, task_id = progress_context if progress_context else (None, None)

    # Single inference pass over all chunks, outer-batched so the progress bar
    # can advance and the underlying call sees full batches on GPU.
    # Sort chunks by length so each batch contains similarly-sized inputs —
    # cuts padding waste dramatically when chunk lengths are uneven.
    n = len(flat_chunks)
    order = sorted(range(n), key=lambda i: len(flat_chunks[i][1]))
    sorted_texts = [flat_chunks[i][1] for i in order]
    sorted_outputs = [None] * n
    for i in range(0, n, batch_size):
        batch = sorted_texts[i : i + batch_size]
        outputs = model.batch_extract_entities(
            batch,
            labels_to_pass,
            batch_size=batch_size,
            threshold=threshold,
            include_spans=True,
        )
        for j, out in enumerate(outputs):
            sorted_outputs[i + j] = out
        if progress and task_id is not None:
            progress.update(task_id, advance=len(batch))

    # Restore original chunk order so doc_chunk_ranges indices line up.
    all_outputs = [None] * n
    for sorted_idx, original_idx in enumerate(order):
        all_outputs[original_idx] = sorted_outputs[sorted_idx]

    # Scatter chunk outputs back to per-document prediction sets.
    for doc_idx, (start, end) in enumerate(doc_chunk_ranges):
        pred_entities = set()
        for chunk_idx in range(start, end):
            _, chunk_text, _offset = flat_chunks[chunk_idx]
            raw_output = all_outputs[chunk_idx]
            if isinstance(raw_output, dict) and "entities" in raw_output:
                for label, items in raw_output["entities"].items():
                    for item in items:
                        surface = chunk_text[item["start"] : item["end"]]
                        pred_entities.add((normalize_entity(surface), label))
            elif isinstance(raw_output, list):
                for item in raw_output:
                    surface = chunk_text[item["start"] : item["end"]]
                    pred_entities.add((normalize_entity(surface), item["label"]))

        gold_entities = gold_per_doc[doc_idx]
        gold_by_label = gold_by_label_per_doc[doc_idx]

        metrics_counts["overall"]["tp"] += len(pred_entities & gold_entities)
        metrics_counts["overall"]["fp"] += len(pred_entities - gold_entities)
        metrics_counts["overall"]["fn"] += len(gold_entities - pred_entities)

        for label in label_keys:
            gold_label = gold_by_label[label]
            pred_label = {e for e in pred_entities if e[1] == label}
            metrics_counts[label]["tp"] += len(pred_label & gold_label)
            metrics_counts[label]["fp"] += len(pred_label - gold_label)
            metrics_counts[label]["fn"] += len(gold_label - pred_label)

    final_scores = {"name": model_name}
    for key, counts in metrics_counts.items():
        final_scores[key] = calculate_metrics(counts["tp"], counts["fp"], counts["fn"])

    return final_scores


def _resolve_gold_tickers(engine, entry):
    """Convert gold annotation spans to the ticker symbols the engine would emit."""
    gold = set()
    text = entry["text"]
    for ann in entry["entities"]:
        span_text = text[ann["start"] : ann["end"]]
        cleaned = engine._clean_token(span_text)
        if not cleaned:
            continue
        if cleaned in engine.valid_tickers:
            gold.add(cleaned)
        else:
            base = cleaned.split()[0]
            resolved = engine.company_to_ticker.get(
                cleaned
            ) or engine.company_to_ticker.get(base)
            if resolved:
                gold.add(resolved)
    return frozenset(gold)


def engine_evaluate_model(adapter_path, dataset, model_name="Model"):
    """Evaluate production F1 by routing through StockRecognizer.recognize_ai().

    Gold is represented as the set of ticker symbols returned by the engine's
    own resolution pipeline on the gold spans, so the comparison is fair:
    both gold and pred are sets of ticker strings, not character offsets.
    """
    from stock_recognizer.engine import StockRecognizer

    engine = StockRecognizer(use_ai=True, adapter_path=adapter_path)

    tp = fp = fn = 0
    for entry in dataset:
        gold_set = _resolve_gold_tickers(engine, entry)
        pred_list = engine.recognize_ai(entry["text"])
        pred_set = frozenset(pred_list)

        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    return {"overall": calculate_metrics(tp, fp, fn)}


def prepare_eval_inputs(dataset, label_keys):
    """Build the per-chunk and per-document structures used by ``evaluate_model``.

    Factored out so the post-training benchmark in ``train.py`` can share
    the exact same chunking/gold logic as the standalone benchmark run.
    """
    flat_chunks = []
    doc_chunk_ranges = []
    gold_per_doc = []
    gold_by_label_per_doc = []

    for doc_idx, entry in enumerate(dataset):
        chunks = chunk_text_for_inference(
            entry["text"], chunk_word_size=150, overlap_words=40
        )
        start = len(flat_chunks)
        for chunk_text, offset in chunks:
            flat_chunks.append((doc_idx, chunk_text, offset))
        doc_chunk_ranges.append((start, len(flat_chunks)))

        # Dedup per document by (normalized surface, label): a ticker mentioned
        # 16× in one post is a single gold key, matching the engine's set output.
        text = entry["text"]
        gold = {
            (normalize_entity(text[e["start"] : e["end"]]), e["label"])
            for e in entry["entities"]
        }
        gold_per_doc.append(gold)
        gold_by_label_per_doc.append(
            {label: {e for e in gold if e[1] == label} for label in label_keys}
        )

    return flat_chunks, doc_chunk_ranges, gold_per_doc, gold_by_label_per_doc


def load_base_model(device=None):
    """Load the base GLiNER2 onto GPU (fp16) when available, else CPU."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GLiNER2.from_pretrained(
        "fastino/gliner2-large-v1",
        map_location=device,
        quantize=(device == "cuda"),
    )
    return model, device


def _metrics_only(scores):
    """Strip the leading 'name' key from ``evaluate_model``'s output so the
    payload we persist matches the cached schema."""
    return {k: v for k, v in scores.items() if k != "name"}


def _load_training_metadata(adapter_path):
    """Look for `training_metadata.json` in the adapter's parent dir.

    Adapter paths from ``locate_adapter_weights`` look like
    ``models/reddit_adapter_v14/best`` or ``.../final``, and the metadata
    sits one level up at ``models/reddit_adapter_v14/training_metadata.json``.
    Returns None silently if the file isn't there (e.g. pre-metadata adapters).
    """
    if not adapter_path:
        return None
    metadata_path = os.path.join(
        os.path.dirname(adapter_path), "training_metadata.json"
    )
    if not os.path.exists(metadata_path):
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _flatten_metadata_into_params(metadata):
    """Flatten the nested metadata dict into the flat `params` schema used by
    the benchmark store + display, while preserving the rich nested fields too."""
    if not metadata:
        return {}
    flat = {}
    cfg = metadata.get("config", {}) or {}
    data = metadata.get("data", {}) or {}
    aug = metadata.get("augmentation", {}) or {}
    negs = metadata.get("negatives") or {}

    # Flat keys that the existing _render_params table expects.
    for k in (
        "num_epochs",
        "batch_size",
        "effective_batch_size",
        "encoder_lr",
        "task_lr",
        "max_grad_norm",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "early_stopping",
        "early_stopping_patience",
        "val_fraction",
    ):
        if cfg.get(k) is not None:
            flat[k] = cfg[k]
    for k in ("train_samples", "val_samples", "test_held_out_tasks"):
        if data.get(k) is not None:
            flat[k] = data[k]
    for k in ("expanded_pool_weight", "cashtag_format_prob", "augmented_files"):
        if aug.get(k) is not None:
            flat[k] = aug[k]
    flat["negatives_count"] = negs.get("tasks", 0) if negs else 0
    for k in ("seed", "git_commit", "timestamp"):
        if metadata.get(k) is not None:
            flat[k] = metadata[k]

    # Keep the nested file inventories accessible — useful for `--show-params`
    # style inspection without trying to flatten lists into the table.
    if data.get("labeled_files"):
        flat["labeled_files"] = data["labeled_files"]
    if data.get("test_files"):
        flat["test_files"] = data["test_files"]
    return flat


def benchmark_adapter(
    name,
    adapter_path,
    training_params=None,
    test_folder=DEFAULT_TEST_FOLDER,
    labels=None,
    base_model=None,
    device=None,
    batch_size=32,
):
    """Evaluate one model (``adapter_path=None`` for clean base), persist
    the result, and return the metrics dict.

    Intended to be called from ``train.py`` directly after training so the
    new adapter's numbers land in the store without a full benchmark sweep.
    """
    if labels is None:
        labels = DEFAULT_LABELS
    label_keys = list(labels.keys()) if isinstance(labels, dict) else labels

    dataset = parse_all_label_studio_exports(test_folder)
    if not dataset:
        raise RuntimeError(f"No annotated tasks found in {test_folder}.")

    test_hash = compute_test_set_hash(dataset)
    store = load_store()
    register_test_set(store, test_hash, dataset, source_folder=test_folder)

    flat_chunks, doc_chunk_ranges, gold_per_doc, gold_by_label_per_doc = (
        prepare_eval_inputs(dataset, label_keys)
    )

    owns_base = base_model is None
    if owns_base:
        base_model, device = load_base_model(device)
    elif device is None:
        device = next(base_model.parameters()).device.type

    if adapter_path and os.path.exists(adapter_path):
        model = copy.deepcopy(base_model)
        model.load_adapter(adapter_path)
    else:
        model = base_model

    scores = evaluate_model(
        model,
        flat_chunks,
        doc_chunk_ranges,
        gold_per_doc,
        gold_by_label_per_doc,
        model_name=name,
        label_descriptions=labels,
        batch_size=batch_size,
    )
    metrics = _metrics_only(scores)

    params = dict(derive_adapter_params(adapter_path)) if adapter_path else {}
    # Pull in the persisted training context (written by train.py) so the
    # benchmark store ends up with full data + config provenance regardless
    # of whether this benchmark was invoked from train.py or standalone.
    metadata = _load_training_metadata(adapter_path) if adapter_path else None
    if metadata:
        params.update(_flatten_metadata_into_params(metadata))
    # An explicit training_params dict still wins (used by tests or callers
    # that want to override what's on disk).
    if training_params:
        params.update(training_params)

    put_result(store, name, test_hash, metrics, params=params)
    save_store(store)

    if model is not base_model:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    return metrics, test_hash


def _render_table(rows):
    """Render a list of ``{"name", "metrics", "params"}`` dicts to a Rich table."""
    table = Table(title="NER Benchmark Breakdown (Per-Entity)", show_lines=False)
    table.add_column("Model Configuration", style="cyan", width=35)
    table.add_column("Entity Type", style="blue")
    table.add_column("Precision (Noise)", justify="right")
    table.add_column("Recall (Detect)", justify="right")
    table.add_column("F1-Score", style="bold magenta", justify="right")

    for row in rows:
        m = row["metrics"]
        suffix = " [dim](cached)[/dim]" if row.get("cached") else ""
        table.add_row(
            f"[bold]{row['name']}[/bold]{suffix}",
            "ticker",
            f"{m['ticker']['p']:.2%}",
            f"{m['ticker']['r']:.2%}",
            f"{m['ticker']['f1']:.2%}",
        )
        table.add_row(
            "",
            "company",
            f"{m['company']['p']:.2%}",
            f"{m['company']['r']:.2%}",
            f"{m['company']['f1']:.2%}",
        )
        table.add_row(
            "",
            "[bold white]OVERALL[/bold white]",
            f"[bold white]{m['overall']['p']:.2%}[/bold white]",
            f"[bold white]{m['overall']['r']:.2%}[/bold white]",
            f"[bold white]{m['overall']['f1']:.2%}[/bold white]",
        )
        table.add_section()
    return table


def _render_params(rows):
    """Side table summarising the non-metric facts we track per adapter."""
    table = Table(title="Adapter parameters", show_lines=False)
    table.add_column("Model", style="cyan", width=35)
    table.add_column("lora_b_norm", justify="right")
    table.add_column("size (KB)", justify="right")
    table.add_column("epochs", justify="right")
    table.add_column("train / val", justify="right")

    for row in rows:
        p = row.get("params") or {}
        train_n = p.get("train_samples")
        val_n = p.get("val_samples")
        split = (
            f"{train_n} / {val_n}" if train_n is not None and val_n is not None else "—"
        )
        table.add_row(
            row["name"],
            f"{p.get('lora_b_norm', '—')}",
            f"{p.get('adapter_size_kb', '—')}",
            f"{p.get('num_epochs', '—')}",
            split,
        )
    return table


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine",
        action="store_true",
        help="Evaluate via StockRecognizer.recognize_ai() instead of raw model",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Re-run NER model inference even if cached results exist. "
        "Use when the model weights or evaluation logic change.",
    )
    parser.add_argument(
        "--no-engine-cache",
        action="store_true",
        help="Re-run the engine evaluation even if cached engine_metrics exist. "
        "Use after changing engine code (AMBIGUOUS_WORDS, resolution logic, etc.). "
        "Much faster than --no-cache: NER inference results are still read from cache.",
    )
    args = parser.parse_args()

    NUM_VERSIONS_TO_TEST = 1  # None = test all adapters
    BATCH_SIZE = 32

    dataset = parse_all_label_studio_exports(DEFAULT_TEST_FOLDER)

    if not dataset:
        console.print("[red]No valid data to evaluate.[/red]")
    else:
        test_hash = compute_test_set_hash(dataset)
        console.print(
            f"Loaded [bold green]{len(dataset)}[/bold green] annotated tasks "
            f"from {DEFAULT_TEST_FOLDER}. test-set hash: [yellow]{test_hash}[/yellow]"
        )

        store = load_store()
        register_test_set(store, test_hash, dataset, source_folder=DEFAULT_TEST_FOLDER)

        label_keys = list(DEFAULT_LABELS.keys())
        flat_chunks, doc_chunk_ranges, gold_per_doc, gold_by_label_per_doc = (
            prepare_eval_inputs(dataset, label_keys)
        )
        console.print(
            f"Prepared [bold green]{len(flat_chunks)}[/bold green] chunks "
            f"across {len(dataset)} documents."
        )

        model_configs = [("Base Model (Clean)", None)]
        available_adapters = get_all_adapters()
        if NUM_VERSIONS_TO_TEST and len(available_adapters) > NUM_VERSIONS_TO_TEST:
            available_adapters = available_adapters[-NUM_VERSIONS_TO_TEST:]
        for adapter in available_adapters:
            model_configs.append((adapter["name"], adapter["path"]))

        # Separate cache hits from the configs we still need to run.
        rows = []
        to_evaluate = []
        for name, adapter_path in model_configs:
            cached = None if args.no_cache else get_cached(store, name, test_hash)
            if cached:
                rows.append(
                    {
                        "name": name,
                        "metrics": cached["metrics"],
                        "params": cached.get("params") or {},
                        "cached": True,
                    }
                )
            else:
                to_evaluate.append((name, adapter_path))

        if to_evaluate:
            shared_base_model, device = load_base_model()
            console.print(
                f"[cyan]Loaded base GLiNER2 onto [bold]{device}[/bold]"
                f"{' (fp16)' if device == 'cuda' else ''}. "
                f"Need to evaluate {len(to_evaluate)}/{len(model_configs)} configs.[/cyan]"
            )

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
            ) as progress:
                overall_task = progress.add_task(
                    "[bold cyan]Overall Evaluation...", total=len(to_evaluate)
                )

                for name, adapter_path in to_evaluate:
                    if adapter_path and os.path.exists(adapter_path):
                        model = copy.deepcopy(shared_base_model)
                        model.load_adapter(adapter_path)
                    else:
                        model = shared_base_model

                    chunk_task = progress.add_task(
                        f"[green]Testing {name}...", total=len(flat_chunks)
                    )
                    scores = evaluate_model(
                        model,
                        flat_chunks,
                        doc_chunk_ranges,
                        gold_per_doc,
                        gold_by_label_per_doc,
                        model_name=name,
                        label_descriptions=DEFAULT_LABELS,
                        batch_size=BATCH_SIZE,
                        progress_context=(progress, chunk_task),
                    )
                    metrics = _metrics_only(scores)
                    params = (
                        dict(derive_adapter_params(adapter_path))
                        if adapter_path
                        else {}
                    )
                    # Same metadata merge as benchmark_adapter() — keeps the
                    # standalone-benchmark code path in sync with the post-train one.
                    metadata = (
                        _load_training_metadata(adapter_path) if adapter_path else None
                    )
                    if metadata:
                        params.update(_flatten_metadata_into_params(metadata))
                    put_result(store, name, test_hash, metrics, params=params)
                    rows.append(
                        {
                            "name": name,
                            "metrics": metrics,
                            "params": params,
                            "cached": False,
                        }
                    )

                    progress.update(overall_task, advance=1)
                    progress.remove_task(chunk_task)
                    if model is not shared_base_model:
                        del model
                        if device == "cuda":
                            torch.cuda.empty_cache()

            save_store(store)

        # Render in the original adapter-version order from model_configs.
        order = [n for n, _ in model_configs]
        rows.sort(key=lambda r: order.index(r["name"]))
        console.print(_render_table(rows))
        console.print(_render_params(rows))

        if args.engine:
            console.print(
                "\n[cyan]Engine evaluation (StockRecognizer.recognize_ai + regex)...[/cyan]"
            )
            eng_table = Table(
                title="Engine Benchmark (production F1)", show_lines=False
            )
            eng_table.add_column("Adapter", style="cyan", width=35)
            eng_table.add_column("Precision", justify="right")
            eng_table.add_column("Recall", justify="right")
            eng_table.add_column("F1", style="bold magenta", justify="right")

            store_dirty = False
            for adapter in available_adapters:
                entry = store["results"].get(adapter["name"], {}).get(test_hash)
                if entry and entry.get("engine_metrics") and not args.no_cache and not args.no_engine_cache:
                    eng_metrics = entry["engine_metrics"]
                    console.print(
                        f"  [dim]{adapter['name']}: using cached engine metrics[/dim]"
                    )
                else:
                    console.print(
                        f"  [dim]Loading engine for {adapter['name']}...[/dim]"
                    )
                    eng_metrics = engine_evaluate_model(
                        adapter["path"], dataset, model_name=adapter["name"]
                    )
                    if entry:
                        entry["engine_metrics"] = eng_metrics
                        store_dirty = True

                m = eng_metrics["overall"]
                eng_table.add_row(
                    adapter["name"],
                    f"{m['p']:.2%}",
                    f"{m['r']:.2%}",
                    f"{m['f1']:.2%}",
                )

            if store_dirty:
                save_store(store)
            console.print(eng_table)

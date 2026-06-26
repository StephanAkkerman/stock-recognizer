"""Break down where a trained adapter actually fails on the held-out test set.

Run:
    python trainer/error_analysis.py                       # latest adapter
    python trainer/error_analysis.py --adapter v4
    python trainer/error_analysis.py --adapter base
    python trainer/error_analysis.py --threshold 0.5 --top 30
    python trainer/error_analysis.py --save-json out.json  # full dump

Scoring is **set-based and deduplicated per document** — the same contract the
engine exposes (``recognize`` returns a *set* of tickers, so an entity is
"caught" once it appears at least once in a post). Entities are keyed by
``(normalized_surface, label)``; repeated mentions and ``$AAPL``/``AAPL`` form
differences collapse to a single key. See ``benchmark.normalize_entity``.

Three error categories are surfaced, aggregated by frequency:

  1. **False positives** — a ``(surface, label)`` the model predicts that is not
     in gold. Highest-leverage signal for hard-negative mining.
  2. **False negatives** — a gold ``(surface, label)`` the model never predicts
     anywhere in the document. Because scoring is deduped, an FN here means the
     model missed the entity in *every* context it appears in — a genuine
     vocabulary/context gap, not a dropped repeat.
  3. **Label confusions** — same normalized surface, different label (e.g.
     "SOFI" predicted as company but gold says ticker). Indicates the
     descriptions for ticker/company aren't separating the two cases.

Plus a per-document hotspot table — top docs by error count.
"""

import argparse
import copy
import json
from collections import Counter, defaultdict

import torch
from rich.console import Console
from rich.table import Table

try:
    from trainer.benchmark import (
        DEFAULT_LABELS,
        DEFAULT_TEST_FOLDER,
        _resolve_gold_tickers,
        get_all_adapters,
        load_base_model,
        normalize_entity,
        parse_all_label_studio_exports,
        prepare_eval_inputs,
    )
except ImportError:
    from benchmark import (
        DEFAULT_LABELS,
        DEFAULT_TEST_FOLDER,
        _resolve_gold_tickers,
        get_all_adapters,
        load_base_model,
        normalize_entity,
        parse_all_label_studio_exports,
        prepare_eval_inputs,
    )

console = Console()


def resolve_adapter(spec):
    """Resolve a --adapter spec ('latest', 'base', 'v4', etc.) to (name, path).

    `base` returns (..., None) so the caller knows to skip adapter loading.
    """
    if spec == "base":
        return ("Base Model (Clean)", None)
    adapters = get_all_adapters()
    if not adapters:
        raise SystemExit("No adapters found under ./models.")
    if spec in (None, "latest"):
        a = adapters[-1]
        return (a["name"], a["path"])
    # Allow "v4" or just "4"
    target = int(spec.lstrip("v"))
    for a in adapters:
        if a["version"] == target:
            return (a["name"], a["path"])
    raise SystemExit(f"Adapter v{target} not found. Available: {[a['version'] for a in adapters]}")


def run_inference(model, flat_chunks, label_descriptions, threshold, batch_size=32):
    """Run batched inference, preserving original chunk order. Mirrors
    `evaluate_model` so error analysis sees identical predictions to benchmark."""
    n = len(flat_chunks)
    order = sorted(range(n), key=lambda i: len(flat_chunks[i][1]))
    sorted_texts = [flat_chunks[i][1] for i in order]
    sorted_outputs = [None] * n
    for i in range(0, n, batch_size):
        batch = sorted_texts[i : i + batch_size]
        outputs = model.batch_extract_entities(
            batch,
            label_descriptions,
            batch_size=batch_size,
            threshold=threshold,
            include_spans=True,
        )
        for j, out in enumerate(outputs):
            sorted_outputs[i + j] = out
    all_outputs = [None] * n
    for sorted_idx, original_idx in enumerate(order):
        all_outputs[original_idx] = sorted_outputs[sorted_idx]
    return all_outputs


def _iter_chunk_items(raw):
    """Yield (start, end, label) for every entity in one chunk's model output."""
    if isinstance(raw, dict) and "entities" in raw:
        for label, items in raw["entities"].items():
            for item in items:
                yield item["start"], item["end"], label
    elif isinstance(raw, list):
        for item in raw:
            yield item["start"], item["end"], item["label"]


def collect_pred_per_doc(all_outputs, flat_chunks, doc_chunk_ranges):
    """Map chunk-local model outputs to a per-document set of (norm_text, label).

    Deduplicated per document to match the engine's set output and
    ``benchmark.evaluate_model`` — repeated mentions collapse to one key.
    """
    pred_per_doc = []
    for _doc_idx, (start, end) in enumerate(doc_chunk_ranges):
        pred = set()
        for chunk_idx in range(start, end):
            _, chunk_text, _offset = flat_chunks[chunk_idx]
            for s, e, label in _iter_chunk_items(all_outputs[chunk_idx]):
                pred.add((normalize_entity(chunk_text[s:e]), label))
        pred_per_doc.append(pred)
    return pred_per_doc


def collect_pred_contexts(all_outputs, flat_chunks, doc_chunk_ranges, context_chars=40):
    """Per-doc dict {(norm_text, label): {"surface", "context"}} from model output.

    The dict is keyed by the normalized surface (so it dedups per document), but
    keeps the *original* surface form and one example context snippet for the
    FP / confusion tables and for downstream span-locating in patch_test_labels.
    """
    out = []
    for _doc_idx, (start, end) in enumerate(doc_chunk_ranges):
        ctx = {}
        for chunk_idx in range(start, end):
            _, chunk_text, _offset = flat_chunks[chunk_idx]
            for s, e, label in _iter_chunk_items(all_outputs[chunk_idx]):
                surface = chunk_text[s:e]
                key = (normalize_entity(surface), label)
                ctx.setdefault(
                    key,
                    {"surface": surface, "context": _make_context(chunk_text, s, e, context_chars)},
                )
        out.append(ctx)
    return out


def collect_gold_contexts(dataset, context_chars=40):
    """Per-doc dict {(norm_text, label): {"surface", "context"}} from gold annotations."""
    out = []
    for entry in dataset:
        text = entry["text"]
        ctx = {}
        for e in entry["entities"]:
            surface = text[e["start"]:e["end"]]
            key = (normalize_entity(surface), e["label"])
            ctx.setdefault(
                key,
                {"surface": surface, "context": _make_context(text, e["start"], e["end"], context_chars)},
            )
        out.append(ctx)
    return out


def categorize_errors(pred_ctx_per_doc, gold_ctx_per_doc, dataset):
    """Split deduped per-document errors into three categories.

    Inputs are the ``{(norm_text, label): context}`` maps from
    ``collect_pred_contexts`` / ``collect_gold_contexts``. Because entities are
    keyed by normalized surface, boundary mismatches (``$AAPL`` vs ``AAPL``)
    collapse away — only pure FP, pure FN, and label confusions remain.

      1. Label confusions — same normalized surface, different label.
      2. Pure FP         — predicted ``(surface, label)`` not in gold.
      3. Pure FN         — gold ``(surface, label)`` the model never predicts.
    """
    pure_fp = []
    pure_fn = []
    confusion = []
    per_doc_counts = []

    for doc_idx, (pred_ctx, gold_ctx) in enumerate(
        zip(pred_ctx_per_doc, gold_ctx_per_doc)
    ):
        pred = set(pred_ctx)
        gold = set(gold_ctx)
        fps = pred - gold
        fns = gold - pred

        # 1. Label confusions: same surface present in both FP and FN pools
        #    under a different label (e.g. SOFI ticker→company).
        fp_labels = defaultdict(set)
        fn_labels = defaultdict(set)
        for surface, label in fps:
            fp_labels[surface].add(label)
        for surface, label in fns:
            fn_labels[surface].add(label)

        consumed_fp = set()
        consumed_fn = set()
        for norm in set(fp_labels) & set(fn_labels):
            for pred_label in fp_labels[norm]:
                for gold_label in fn_labels[norm]:
                    if pred_label == gold_label:
                        continue
                    src = gold_ctx.get((norm, gold_label)) or pred_ctx.get((norm, pred_label))
                    confusion.append({
                        "doc_idx": doc_idx,
                        "text": src["surface"],
                        "gold_label": gold_label,
                        "pred_label": pred_label,
                        "context": src["context"],
                    })
                    consumed_fp.add((norm, pred_label))
                    consumed_fn.add((norm, gold_label))

        # 2 & 3. Everything unconsumed is a pure FP / FN.
        for key in fps - consumed_fp:
            pure_fp.append({
                "doc_idx": doc_idx,
                "text": pred_ctx[key]["surface"],
                "label": key[1],
                "context": pred_ctx[key]["context"],
            })
        for key in fns - consumed_fn:
            pure_fn.append({
                "doc_idx": doc_idx,
                "text": gold_ctx[key]["surface"],
                "label": key[1],
                "context": gold_ctx[key]["context"],
            })

        per_doc_counts.append({
            "doc_idx": doc_idx,
            "n_errors": len(fps) + len(fns),
            "n_fp": len(fps),
            "n_fn": len(fns),
            "preview": dataset[doc_idx]["text"][:80].replace("\n", " "),
        })

    return {
        "pure_fp": pure_fp,
        "pure_fn": pure_fn,
        "confusion": confusion,
        "per_doc": per_doc_counts,
    }


def _make_context(text, start, end, n_chars):
    """Return a window of text around (start, end) with the entity highlighted."""
    left = max(0, start - n_chars)
    right = min(len(text), end + n_chars)
    prefix = "..." if left > 0 else ""
    suffix = "..." if right < len(text) else ""
    snippet = (
        text[left:start] + "[" + text[start:end] + "]" + text[end:right]
    ).replace("\n", " ")
    return f"{prefix}{snippet}{suffix}"


def _aggregate(records, key_fields, top):
    """Bucket records by `key_fields`, return [(key_tuple, count, example_record), ...]."""
    counter = Counter()
    examples = {}
    for r in records:
        key = tuple(r[f] for f in key_fields)
        counter[key] += 1
        examples.setdefault(key, r)
    n = top if (top and top > 0) else None  # None / 0 → return everything
    return [(key, count, examples[key]) for key, count in counter.most_common(n)]


def render_fp_fn_table(title, agg, label_col):
    """Render a top-N FP or FN table."""
    table = Table(title=title, show_lines=False)
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("text", style="bold")
    table.add_column(label_col)
    table.add_column("example context")
    for key, count, ex in agg:
        text, label = key
        table.add_row(str(count), text, label, ex["context"])
    return table


def render_confusion_table(agg):
    table = Table(title="Label confusions (same surface, different label)", show_lines=False)
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("text", style="bold")
    table.add_column("gold→pred")
    table.add_column("example context")
    for key, count, ex in agg:
        text, gold_label, pred_label = key
        table.add_row(str(count), text, f"{gold_label}→{pred_label}", ex["context"])
    return table


def _find_ticker_context(text, ticker, context_chars):
    """Return a context snippet around the first occurrence of ticker in text."""
    import re
    m = re.search(rf"\$?{re.escape(ticker)}\b", text, re.IGNORECASE)
    if m:
        return _make_context(text, m.start(), m.end(), context_chars)
    return text[: context_chars * 2].replace("\n", " ")


def engine_categorize_errors(dataset, adapter_path, context_chars=40):
    """Run the full engine pipeline and surface FP/FN at the resolved-ticker level.

    Uses StockRecognizer.recognize_ai() for predictions and the same
    _resolve_gold_tickers logic as benchmark.engine_evaluate_model so the
    numbers match exactly.

    Returns (fp_records, fn_records, per_doc, summary) where summary is a dict
    with tp/fp/fn/p/r/f1 aggregated across all documents.
    """
    from stock_recognizer.engine import StockRecognizer

    engine = StockRecognizer(use_ai=True, adapter_path=adapter_path)

    fp_records, fn_records, per_doc = [], [], []
    total_tp = total_fp = total_fn = 0
    for doc_idx, entry in enumerate(dataset):
        text = entry["text"]
        gold_set = _resolve_gold_tickers(engine, entry)
        pred_set = frozenset(engine.recognize_ai(text))

        fps = pred_set - gold_set
        fns = gold_set - pred_set
        tp = len(pred_set & gold_set)
        total_tp += tp
        total_fp += len(fps)
        total_fn += len(fns)

        for ticker in sorted(fps):
            fp_records.append({
                "doc_idx": doc_idx,
                "text": ticker,
                "label": "ticker",
                "context": _find_ticker_context(text, ticker, context_chars),
            })
        for ticker in sorted(fns):
            fn_records.append({
                "doc_idx": doc_idx,
                "text": ticker,
                "label": "ticker",
                "context": _find_ticker_context(text, ticker, context_chars),
            })
        per_doc.append({
            "doc_idx": doc_idx,
            "n_errors": len(fps) + len(fns),
            "n_fp": len(fps),
            "n_fn": len(fns),
            "preview": text[:80].replace("\n", " "),
        })

    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    summary = {"tp": total_tp, "fp": total_fp, "fn": total_fn, "p": p, "r": r, "f1": f1}
    return fp_records, fn_records, per_doc, summary


def render_doc_hotspots(per_doc, top):
    n = top if (top and top > 0) else None
    table = Table(title=f"Top {n or 'all'} documents by error count", show_lines=False)
    table.add_column("doc", justify="right", style="dim", width=5)
    table.add_column("errors", justify="right")
    table.add_column("FP", justify="right", style="yellow")
    table.add_column("FN", justify="right", style="red")
    table.add_column("preview")
    sorted_docs = sorted(per_doc, key=lambda d: d["n_errors"], reverse=True)[:n]
    for d in sorted_docs:
        if d["n_errors"] == 0:
            continue
        table.add_row(
            str(d["doc_idx"]),
            str(d["n_errors"]),
            str(d["n_fp"]),
            str(d["n_fn"]),
            d["preview"],
        )
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="latest",
                        help="'latest' (default), 'base', or a version like 'v4' / '4'.")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Confidence threshold (default 0.75, matches benchmark).")
    parser.add_argument("--top", type=int, default=50,
                        help="Max rows per error table (default 50; use 0 to show all).")
    parser.add_argument("--context", type=int, default=40,
                        help="Chars of context on each side of an error (default 40).")
    parser.add_argument("--test-folder", default=DEFAULT_TEST_FOLDER,
                        help=f"Held-out test set folder (default {DEFAULT_TEST_FOLDER}).")
    parser.add_argument("--save-json", default=None,
                        help="Optional path to dump the full categorised error data as JSON.")
    parser.add_argument("--engine", action="store_true",
                        help="Analyse the full StockRecognizer.recognize_ai() pipeline "
                             "instead of the raw NER model. FPs/FNs are at the resolved-ticker "
                             "level. No GPU load — uses cached engine.")
    args = parser.parse_args()

    adapter_name, adapter_path = resolve_adapter(args.adapter)
    dataset = parse_all_label_studio_exports(args.test_folder)
    if not dataset:
        console.print(f"[red]No data in {args.test_folder}.[/red]")
        raise SystemExit(1)

    top_label = args.top if args.top and args.top > 0 else "all"

    # ── Engine mode: full StockRecognizer pipeline, no GPU needed ──────────────
    if args.engine:
        console.print(
            f"[cyan]Engine error analysis[/cyan]: [bold]{adapter_name}[/bold]\n"
            f"Test set: [bold green]{len(dataset)}[/bold green] docs"
        )
        fp_records, fn_records, per_doc, summary = engine_categorize_errors(
            dataset, adapter_path, args.context
        )
        console.print(
            f"\n[bold]TP={summary['tp']}  FP={summary['fp']}  FN={summary['fn']}[/bold]   "
            f"P={summary['p']:.2%}  R={summary['r']:.2%}  F1={summary['f1']:.2%}   "
            f"[dim](resolved-ticker level)[/dim]\n"
        )
        if fp_records:
            agg = _aggregate(fp_records, ("text", "label"), args.top)
            console.print(render_fp_fn_table(
                f"Top {top_label} engine false positives", agg, "label"))
        if fn_records:
            agg = _aggregate(fn_records, ("text", "label"), args.top)
            console.print(render_fp_fn_table(
                f"Top {top_label} engine false negatives", agg, "label"))
        console.print(render_doc_hotspots(per_doc, args.top))
        if args.save_json:
            with open(args.save_json, "w", encoding="utf-8") as f:
                json.dump(
                    {"adapter": adapter_name, "mode": "engine",
                     "summary": summary, "fp": fp_records, "fn": fn_records},
                    f, indent=2, ensure_ascii=False,
                )
            console.print(f"[green]Written to {args.save_json}[/green]")
        return

    # ── NER model mode: raw model output vs. gold spans ────────────────────────
    label_keys = list(DEFAULT_LABELS.keys())
    flat_chunks, doc_chunk_ranges, _, _ = prepare_eval_inputs(dataset, label_keys)
    gold_ctx_per_doc = collect_gold_contexts(dataset, args.context)
    total_gold = sum(len(g) for g in gold_ctx_per_doc)
    console.print(
        f"Adapter: [bold cyan]{adapter_name}[/bold cyan] @ threshold={args.threshold}\n"
        f"Test set: [bold green]{len(dataset)}[/bold green] docs, "
        f"[bold green]{total_gold}[/bold green] unique gold entities (deduped per doc), "
        f"[bold green]{len(flat_chunks)}[/bold green] chunks"
    )

    base_model, device = load_base_model()
    if adapter_path:
        model = copy.deepcopy(base_model)
        model.load_adapter(adapter_path)
    else:
        model = base_model

    all_outputs = run_inference(model, flat_chunks, DEFAULT_LABELS, args.threshold)
    pred_ctx_per_doc = collect_pred_contexts(
        all_outputs, flat_chunks, doc_chunk_ranges, args.context
    )
    total_pred = sum(len(p) for p in pred_ctx_per_doc)
    n_tp = sum(
        len(set(p) & set(g))
        for p, g in zip(pred_ctx_per_doc, gold_ctx_per_doc)
    )

    categories = categorize_errors(pred_ctx_per_doc, gold_ctx_per_doc, dataset)
    n_fp = total_pred - n_tp
    n_fn = total_gold - n_tp
    p = n_tp / total_pred if total_pred else 0.0
    r = n_tp / total_gold if total_gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0

    console.print(
        f"\n[bold]TP={n_tp}  FP={n_fp}  FN={n_fn}[/bold]   "
        f"P={p:.2%}  R={r:.2%}  F1={f1:.2%}   "
        f"[dim](set-based, deduped per document)[/dim]"
    )
    console.print(
        f"  pure FP: {len(categories['pure_fp'])}  |  "
        f"pure FN: {len(categories['pure_fn'])}  |  "
        f"label confusion: {len(categories['confusion'])}\n"
    )

    if categories["pure_fp"]:
        agg = _aggregate(categories["pure_fp"], ("text", "label"), args.top)
        console.print(render_fp_fn_table(
            f"Top {top_label} false positives (hallucinated entities)", agg, "label"))
    if categories["pure_fn"]:
        agg = _aggregate(categories["pure_fn"], ("text", "label"), args.top)
        console.print(render_fp_fn_table(
            f"Top {top_label} false negatives (missed entities)", agg, "label"))
    if categories["confusion"]:
        agg = _aggregate(categories["confusion"], ("text", "gold_label", "pred_label"), args.top)
        console.print(render_confusion_table(agg))

    console.print(render_doc_hotspots(categories["per_doc"], args.top))

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "adapter": adapter_name,
                    "threshold": args.threshold,
                    "test_folder": args.test_folder,
                    "summary": {"tp": n_tp, "fp": n_fp, "fn": n_fn, "p": p, "r": r, "f1": f1},
                    "categories": categories,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        console.print(f"[green]Full error data written to {args.save_json}[/green]")

    if device == "cuda" and adapter_path:
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

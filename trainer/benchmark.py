import copy
import glob
import json
import os
import re

from gliner2 import GLiNER2
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

console = Console()


def get_all_adapters(models_dir="./models"):
    """Scans the models directory and returns a sorted list of all valid adapters."""
    if not os.path.exists(models_dir):
        return []

    adapters = glob.glob(os.path.join(models_dir, "reddit_adapter*"))
    valid_adapters = []

    for adapter_dir in adapters:
        final_path = os.path.join(adapter_dir, "final")
        if not os.path.exists(final_path):
            continue

        folder_name = os.path.basename(adapter_dir)

        # Extract version number
        match = re.search(r"_v(\d+)$", folder_name)
        if match:
            v = int(match.group(1))
        elif folder_name == "reddit_adapter":
            v = 1
        else:
            continue

        valid_adapters.append(
            {"version": v, "name": f"GLiNER2 Large + Adapter v{v}", "path": final_path}
        )

    # Sort by version number ascending
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
    model_instance,
    test_data,
    model_name="Model",
    label_descriptions=None,
    adapter_path=None,
    progress_context=None,  # <--- NEW PARAMETER
):
    """Calculates NER metrics using a pre-instantiated model, batching chunks together."""

    model = model_instance
    if adapter_path and os.path.exists(adapter_path):
        model.load_adapter(adapter_path)

    labels_to_pass = label_descriptions if label_descriptions else ["ticker", "company"]
    label_keys = (
        list(labels_to_pass.keys())
        if isinstance(labels_to_pass, dict)
        else labels_to_pass
    )

    metrics_counts = {k: {"tp": 0, "fp": 0, "fn": 0} for k in label_keys}
    metrics_counts["overall"] = {"tp": 0, "fp": 0, "fn": 0}

    # Unpack the progress bar tools if they were passed in
    progress, task_id = progress_context if progress_context else (None, None)

    for entry in test_data:
        text = entry["text"]
        gold_entities = {(e["start"], e["end"], e["label"]) for e in entry["entities"]}
        pred_entities = set()

        chunks = chunk_text_for_inference(text, chunk_word_size=150, overlap_words=40)
        chunk_texts = [c[0] for c in chunks]

        batch_outputs = model.batch_extract_entities(
            chunk_texts, labels_to_pass, threshold=0.75, include_spans=True
        )

        for (chunk_text, chunk_char_offset), raw_output in zip(chunks, batch_outputs):
            if isinstance(raw_output, dict) and "entities" in raw_output:
                for label, items in raw_output["entities"].items():
                    for item in items:
                        abs_start = chunk_char_offset + item["start"]
                        abs_end = chunk_char_offset + item["end"]
                        pred_entities.add((abs_start, abs_end, label))
            elif isinstance(raw_output, list):
                for item in raw_output:
                    abs_start = chunk_char_offset + item["start"]
                    abs_end = chunk_char_offset + item["end"]
                    pred_entities.add((abs_start, abs_end, item["label"]))

        # Calculate Overall Metrics
        metrics_counts["overall"]["tp"] += len(pred_entities & gold_entities)
        metrics_counts["overall"]["fp"] += len(pred_entities - gold_entities)
        metrics_counts["overall"]["fn"] += len(gold_entities - pred_entities)

        # Calculate Per-Label Metrics
        for label in label_keys:
            gold_label = {e for e in gold_entities if e[2] == label}
            pred_label = {e for e in pred_entities if e[2] == label}

            metrics_counts[label]["tp"] += len(pred_label & gold_label)
            metrics_counts[label]["fp"] += len(pred_label - gold_label)
            metrics_counts[label]["fn"] += len(gold_label - pred_label)

        # --- UPDATE THE PROGRESS BAR PER DOCUMENT ---
        if progress and task_id is not None:
            progress.update(task_id, advance=1)

    final_scores = {"name": model_name}
    for key, counts in metrics_counts.items():
        final_scores[key] = calculate_metrics(counts["tp"], counts["fp"], counts["fn"])

    return final_scores


if __name__ == "__main__":
    # --- CUSTOMIZATION SETTINGS ---
    NUM_VERSIONS_TO_TEST = 2
    # ------------------------------

    ls_export_folder = "data/labeled"
    dataset = parse_all_label_studio_exports(ls_export_folder)

    labels = {
        "ticker": "A stock market ticker symbol, usually 1-5 letters, often preceded by a dollar sign (e.g., $AAPL, TSLA). MUST NOT be option strikes (e.g., 140c), prices, index names, or internet slang acronyms (e.g., NFA, Lmfao, JPOW).",
        "company": "The name of a corporation, hedge fund, or business entity (e.g., Microsoft, Melvin Capital, Valve). MUST NOT be an uppercase ticker symbol, an index (e.g., Dow Jones), or generic finance terms.",
    }

    if not dataset:
        console.print("[red]No valid data to evaluate.[/red]")
    else:
        console.print(
            f"Loaded [bold green]{len(dataset)}[/bold green] annotated tasks from {ls_export_folder}."
        )
        results = []

        # SPEED OPTIMIZATION: Instantiate base model EXACTLY ONCE globally
        console.print("[cyan]Loading base GLiNER2 architecture into memory...[/cyan]")
        shared_base_model = GLiNER2.from_pretrained("fastino/gliner2-large-v1")

        # Set up evaluation targets
        model_configs = [
            ("Base Model (Clean)", None),
        ]

        available_adapters = get_all_adapters()
        if NUM_VERSIONS_TO_TEST and len(available_adapters) > NUM_VERSIONS_TO_TEST:
            available_adapters = available_adapters[-NUM_VERSIONS_TO_TEST:]

        for adapter in available_adapters:
            model_configs.append((adapter["name"], adapter["path"]))

        # We add some extra formatting columns to make the progress bar look great
        from rich.progress import (
            BarColumn,
            TaskProgressColumn,
            TextColumn,
            TimeRemainingColumn,
        )

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:

            # 1. The overall tracker (e.g., 0/3 Models)
            overall_task = progress.add_task(
                "[bold cyan]Overall Evaluation...", total=len(model_configs)
            )

            for name, adapter_path in model_configs:
                # 2. The granular tracker (e.g., 0/457 Documents)
                doc_task = progress.add_task(
                    f"[green]Testing {name}...", total=len(dataset)
                )

                model_copy = (
                    copy.deepcopy(shared_base_model)
                    if adapter_path
                    else shared_base_model
                )

                results.append(
                    evaluate_model(
                        model_copy,
                        dataset,
                        name,
                        label_descriptions=labels,
                        adapter_path=adapter_path,
                        progress_context=(progress, doc_task),
                    )
                )

                # Advance the overall model tracker and remove the finished document tracker
                progress.update(overall_task, advance=1)
                progress.remove_task(doc_task)

        # Output Table
        table = Table(title="NER Benchmark Breakdown (Per-Entity)", show_lines=False)
        table.add_column("Model Configuration", style="cyan", width=35)
        table.add_column("Entity Type", style="blue")
        table.add_column("Precision (Noise)", justify="right")
        table.add_column("Recall (Detect)", justify="right")
        table.add_column("F1-Score", style="bold magenta", justify="right")

        for r in results:
            table.add_row(
                f"[bold]{r['name']}[/bold]",
                "ticker",
                f"{r['ticker']['p']:.2%}",
                f"{r['ticker']['r']:.2%}",
                f"{r['ticker']['f1']:.2%}",
            )
            table.add_row(
                "",
                "company",
                f"{r['company']['p']:.2%}",
                f"{r['company']['r']:.2%}",
                f"{r['company']['f1']:.2%}",
            )
            table.add_row(
                "",
                "[bold white]OVERALL[/bold white]",
                f"[bold white]{r['overall']['p']:.2%}[/bold white]",
                f"[bold white]{r['overall']['r']:.2%}[/bold white]",
                f"[bold white]{r['overall']['f1']:.2%}[/bold white]",
            )
            table.add_section()

        console.print(table)

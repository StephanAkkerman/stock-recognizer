import glob
import json
import os

from gliner2 import GLiNER2
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

console = Console()


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


def evaluate_model(
    base_model_path,
    test_data,
    model_name="Model",
    label_descriptions=None,
    adapter_path=None,
):
    """Calculates NER metrics, broken down by entity type."""
    is_local = os.path.isdir(base_model_path)
    model = GLiNER2.from_pretrained(base_model_path, local_files_only=is_local)

    if adapter_path:
        if os.path.exists(adapter_path):
            model.load_adapter(adapter_path)
        else:
            console.print(
                f"[bold yellow]Warning: Adapter at {adapter_path} not found![/bold yellow]"
            )

    labels_to_pass = label_descriptions if label_descriptions else ["ticker", "company"]
    label_keys = (
        list(labels_to_pass.keys())
        if isinstance(labels_to_pass, dict)
        else labels_to_pass
    )

    # Track metrics separately for each label, plus an overall bucket
    metrics_counts = {k: {"tp": 0, "fp": 0, "fn": 0} for k in label_keys}
    metrics_counts["overall"] = {"tp": 0, "fp": 0, "fn": 0}

    for entry in test_data:
        text = entry["text"]
        gold_entities = {(e["start"], e["end"], e["label"]) for e in entry["entities"]}

        raw_output = model.extract_entities(
            text, labels_to_pass, threshold=0.75, include_spans=True
        )

        flat_predictions = []
        if isinstance(raw_output, dict) and "entities" in raw_output:
            for label, items in raw_output["entities"].items():
                for item in items:
                    flat_predictions.append(
                        {"start": item["start"], "end": item["end"], "label": label}
                    )

        pred_entities = {(p["start"], p["end"], p["label"]) for p in flat_predictions}

        # 1. Calculate Overall Metrics
        metrics_counts["overall"]["tp"] += len(pred_entities & gold_entities)
        metrics_counts["overall"]["fp"] += len(pred_entities - gold_entities)
        metrics_counts["overall"]["fn"] += len(gold_entities - pred_entities)

        # 2. Calculate Per-Label Metrics
        for label in label_keys:
            gold_label = {e for e in gold_entities if e[2] == label}
            pred_label = {e for e in pred_entities if e[2] == label}

            metrics_counts[label]["tp"] += len(pred_label & gold_label)
            metrics_counts[label]["fp"] += len(pred_label - gold_label)
            metrics_counts[label]["fn"] += len(gold_label - pred_label)

    # Compile final scores
    final_scores = {"name": model_name}
    for key, counts in metrics_counts.items():
        final_scores[key] = calculate_metrics(counts["tp"], counts["fp"], counts["fn"])

    return final_scores


if __name__ == "__main__":
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

        model_configs = [
            ("fastino/gliner2-large-v1", "GLiNER2 Large Base", None),
            (
                "fastino/gliner2-large-v1",
                "GLiNER2 Large + Adapter v1",
                "./models/reddit_adapter/final",
            ),
            (
                "fastino/gliner2-large-v1",
                "GLiNER2 Large + Adapter v2",
                "./models/reddit_adapter_v2/final",
            ),
        ]

        with Progress() as progress:
            task = progress.add_task(
                "[cyan]Evaluating models...", total=len(model_configs)
            )

            for base_path, name, adapter_path in model_configs:
                results.append(
                    evaluate_model(
                        base_path,
                        dataset,
                        name,
                        label_descriptions=labels,
                        adapter_path=adapter_path,
                    )
                )
                progress.update(task, advance=1)

        # Output Table (Multi-row per model)
        table = Table(title="NER Benchmark Breakdown (Per-Entity)", show_lines=False)
        table.add_column("Model Configuration", style="cyan", width=35)
        table.add_column("Entity Type", style="blue")
        table.add_column("Precision (Noise)", justify="right")
        table.add_column("Recall (Detect)", justify="right")
        table.add_column("F1-Score", style="bold magenta", justify="right")

        for r in results:
            # Row 1: Ticker
            table.add_row(
                f"[bold]{r['name']}[/bold]",
                "ticker",
                f"{r['ticker']['p']:.2%}",
                f"{r['ticker']['r']:.2%}",
                f"{r['ticker']['f1']:.2%}",
            )
            # Row 2: Company
            table.add_row(
                "",
                "company",
                f"{r['company']['p']:.2%}",
                f"{r['company']['r']:.2%}",
                f"{r['company']['f1']:.2%}",
            )
            # Row 3: Overall
            table.add_row(
                "",
                "[bold white]OVERALL[/bold white]",
                f"[bold white]{r['overall']['p']:.2%}[/bold white]",
                f"[bold white]{r['overall']['r']:.2%}[/bold white]",
                f"[bold white]{r['overall']['f1']:.2%}[/bold white]",
            )
            # Add a line separating the models
            table.add_section()

        console.print(table)

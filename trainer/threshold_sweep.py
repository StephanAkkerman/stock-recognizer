"""Sweep the decision threshold across all adapters on the held-out test set.

Run:
    python trainer/threshold_sweep.py

Prints one Rich table per adapter showing precision / recall / F1 at each
threshold, with the F1-optimal row marked. Intended for picking the operating
point after `benchmark.py` has selected an adapter version.

Why this exists: a less-overfit adapter has flatter confidence distributions
and is unfairly penalised by the default 0.75 threshold that older adapters
were tuned against. Sweeping is the cheap way to compare versions at each
model's own best operating point.
"""

import copy
import os

import torch
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
    from trainer.benchmark import (
        DEFAULT_LABELS,
        DEFAULT_TEST_FOLDER,
        evaluate_model,
        get_all_adapters,
        load_base_model,
        parse_all_label_studio_exports,
        prepare_eval_inputs,
    )
except ImportError:
    from benchmark import (
        DEFAULT_LABELS,
        DEFAULT_TEST_FOLDER,
        evaluate_model,
        get_all_adapters,
        load_base_model,
        parse_all_label_studio_exports,
        prepare_eval_inputs,
    )

console = Console()

# Inclusive range — 0.30 to 0.80 in 0.05 steps. Below 0.30 noise dominates;
# above 0.80 every model collapses.
THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(11)]
BATCH_SIZE = 32


def sweep_adapter(
    model,
    flat_chunks,
    doc_chunk_ranges,
    gold_per_doc,
    gold_by_label_per_doc,
    thresholds,
    progress_ctx=None,
):
    """Returns list of (threshold, scores_dict) for one model across all thresholds."""
    rows = []
    progress, task_id = progress_ctx if progress_ctx else (None, None)
    for t in thresholds:
        scores = evaluate_model(
            model,
            flat_chunks,
            doc_chunk_ranges,
            gold_per_doc,
            gold_by_label_per_doc,
            label_descriptions=DEFAULT_LABELS,
            batch_size=BATCH_SIZE,
            threshold=t,
        )
        rows.append((t, scores))
        if progress and task_id is not None:
            progress.update(task_id, advance=1)
    return rows


def _render(name, rows):
    table = Table(title=f"{name} — threshold sweep", show_lines=False)
    table.add_column("threshold", justify="right")
    table.add_column("ticker F1", justify="right")
    table.add_column("company F1", justify="right")
    table.add_column("P", justify="right")
    table.add_column("R", justify="right")
    table.add_column("overall F1", style="bold magenta", justify="right")

    best_f1 = max(r[1]["overall"]["f1"] for r in rows)
    for t, scores in rows:
        o = scores["overall"]
        marker = " [bold green]*[/bold green]" if abs(o["f1"] - best_f1) < 1e-9 else ""
        table.add_row(
            f"{t:.2f}",
            f"{scores['ticker']['f1']:.2%}",
            f"{scores['company']['f1']:.2%}",
            f"{o['p']:.2%}",
            f"{o['r']:.2%}",
            f"{o['f1']:.2%}{marker}",
        )
    return table


if __name__ == "__main__":
    dataset = parse_all_label_studio_exports(DEFAULT_TEST_FOLDER)
    if not dataset:
        console.print(f"[red]No test data in {DEFAULT_TEST_FOLDER}.[/red]")
        raise SystemExit(1)

    label_keys = list(DEFAULT_LABELS.keys())
    flat_chunks, doc_chunk_ranges, gold_per_doc, gold_by_label_per_doc = (
        prepare_eval_inputs(dataset, label_keys)
    )
    console.print(
        f"Test set: [bold green]{len(dataset)}[/bold green] docs, "
        f"[bold green]{len(flat_chunks)}[/bold green] chunks, "
        f"thresholds: {THRESHOLDS[0]}..{THRESHOLDS[-1]} ({len(THRESHOLDS)} steps)"
    )

    base_model, device = load_base_model()
    console.print(
        f"[cyan]Base GLiNER2 loaded on [bold]{device}[/bold]"
        f"{' (fp16)' if device == 'cuda' else ''}.[/cyan]"
    )

    configs = [("Base Model (Clean)", None)]
    for a in get_all_adapters():
        configs.append((a["name"], a["path"]))

    all_results = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        overall_task = progress.add_task(
            "[bold cyan]Sweeping...", total=len(configs) * len(THRESHOLDS)
        )
        for name, adapter_path in configs:
            if adapter_path and os.path.exists(adapter_path):
                model = copy.deepcopy(base_model)
                model.load_adapter(adapter_path)
            else:
                model = base_model

            rows = sweep_adapter(
                model,
                flat_chunks,
                doc_chunk_ranges,
                gold_per_doc,
                gold_by_label_per_doc,
                THRESHOLDS,
                progress_ctx=(progress, overall_task),
            )
            all_results.append((name, rows))

            if model is not base_model:
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()

    for name, rows in all_results:
        console.print(_render(name, rows))

    # Compact summary: best threshold per adapter, sorted by F1.
    summary = Table(title="Best-F1 operating point per adapter", show_lines=False)
    summary.add_column("Model", style="cyan")
    summary.add_column("best threshold", justify="right")
    summary.add_column("P", justify="right")
    summary.add_column("R", justify="right")
    summary.add_column("F1", style="bold magenta", justify="right")
    ranked = []
    for name, rows in all_results:
        best_t, best_scores = max(rows, key=lambda r: r[1]["overall"]["f1"])
        ranked.append((name, best_t, best_scores["overall"]))
    ranked.sort(key=lambda r: r[2]["f1"], reverse=True)
    for name, t, o in ranked:
        summary.add_row(
            name, f"{t:.2f}", f"{o['p']:.2%}", f"{o['r']:.2%}", f"{o['f1']:.2%}"
        )
    console.print(summary)

"""Build a stratified held-out test set from every labeled file.

Samples ~15% of valid annotated tasks from each `data/labeled/labeled_*.json`
using a deterministic seed, writes them to `data/test/`. The training loader
already excludes test task IDs from train+val (and from any augmented copies
sharing those IDs) via the `test_folder` argument to `parse_all_labeled_data`,
so we do NOT delete the sampled tasks from their source files — the same
task can sit in both folders and the loader dedups by ID.

Why this exists: the previous test set was `labeled_300.json` whole-file,
which has a very different distribution from the training mix:

  - 1.85 entities/task vs. 3-10 elsewhere (test was the easiest file)
  - 32% cashtag rate vs. 8-20% in training (cashtag-heavy test, bare-heavy train)
  - 254 chars/post vs. 376-2062 elsewhere (short posts vs. long DDs)

This made cross-version comparisons noisy and over-represented cashtag-recall
in the score. A stratified sample fixes both: distribution-matched test, and
the entity-volume balance reflects production usage.

Run:
    python trainer/split_test_set.py            # writes test/, prints summary
    python trainer/split_test_set.py --seed 7   # different deterministic split

After running, re-benchmark every adapter (`python trainer/benchmark.py`)
because the new test set has a different hash — old cached results don't
apply.
"""

import argparse
import glob
import json
import os
import random
import shutil

from rich.console import Console
from rich.table import Table

console = Console()

LABELED_FOLDER = "data/labeled"
TEST_FOLDER = "data/test"


def _count_entities(task):
    return sum(
        1
        for r in task.get("annotations", [{}])[0].get("result", [])
        if r.get("type") == "labels"
    )


def _count_cashtags(task):
    text = task.get("data", {}).get("text", "")
    n = 0
    for r in task["annotations"][0].get("result", []):
        if r.get("type") != "labels":
            continue
        if r["value"]["labels"][0] != "ticker":
            continue
        s, e = r["value"]["start"], r["value"]["end"]
        if text[s:e].startswith("$"):
            n += 1
    return n


def restore_legacy_test_files(labeled_folder, test_folder):
    """Move any non-stratified labeled_*.json out of test/ back to labeled/.

    The first-pass test split moved `labeled_300.json` whole into `data/test/`.
    For the stratified split we want it back in labeled/, with only a fraction
    held out. Anything matching the original naming (`labeled_*.json`, not
    `test_labeled_*.json`) gets restored.
    """
    if not os.path.isdir(test_folder):
        return
    for fp in glob.glob(os.path.join(test_folder, "labeled_*.json")):
        name = os.path.basename(fp)
        if name.startswith("test_"):
            continue
        target = os.path.join(labeled_folder, name)
        if os.path.exists(target):
            console.print(
                f"[yellow]Skipping restore of {name}: already exists in labeled/[/yellow]"
            )
            continue
        shutil.move(fp, target)
        console.print(f"[cyan]Restored {fp} -> {target}[/cyan]")


def clear_existing_test_files(test_folder):
    """Wipe prior test_*.json so stale splits don't accumulate."""
    if not os.path.isdir(test_folder):
        return
    for fp in glob.glob(os.path.join(test_folder, "test_*.json")):
        os.remove(fp)


def stratified_split(labeled_folder, test_folder, test_fraction, seed):
    """For each source file, sample `test_fraction` of tasks by ID."""
    os.makedirs(test_folder, exist_ok=True)
    rng = random.Random(seed)

    summary = []
    for fp in sorted(glob.glob(os.path.join(labeled_folder, "*.json"))):
        name = os.path.basename(fp)
        # Skip mined negatives, backups, and any unrelated artifact files.
        if "negatives" in name or name.endswith(".bak"):
            continue

        with open(fp, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        valid = [
            t
            for t in tasks
            if t.get("annotations")
            and not t["annotations"][0].get("was_cancelled")
            and _count_entities(t) > 0
        ]
        if not valid:
            continue

        ids = sorted({t["id"] for t in valid})
        rng.shuffle(ids)
        n_test = max(1, int(round(len(ids) * test_fraction)))
        test_ids = set(ids[:n_test])
        test_tasks = [t for t in valid if t["id"] in test_ids]

        out_path = os.path.join(test_folder, f"test_{name}")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(test_tasks, f, indent=2, ensure_ascii=False)

        summary.append(
            {
                "file": name,
                "total_tasks": len(valid),
                "test_tasks": len(test_tasks),
                "test_entities": sum(_count_entities(t) for t in test_tasks),
                "test_cashtags": sum(_count_cashtags(t) for t in test_tasks),
            }
        )
    return summary


DEFAULT_SEED = 42
DEFAULT_TEST_FRACTION = 0.15


def _render_summary(summary, seed, test_fraction):
    table = Table(
        title=f"Stratified test split — {test_fraction:.0%} per file, seed={seed}"
    )
    table.add_column("file")
    table.add_column("source tasks", justify="right")
    table.add_column("test tasks", justify="right")
    table.add_column("test entities", justify="right")
    table.add_column("test cashtags", justify="right")
    table.add_column("cashtag%", justify="right")

    totals = {"src": 0, "tasks": 0, "ents": 0, "cash": 0}
    for row in summary:
        totals["src"] += row["total_tasks"]
        totals["tasks"] += row["test_tasks"]
        totals["ents"] += row["test_entities"]
        totals["cash"] += row["test_cashtags"]
        cash_pct = (
            row["test_cashtags"] / row["test_entities"] * 100
            if row["test_entities"]
            else 0.0
        )
        table.add_row(
            row["file"],
            str(row["total_tasks"]),
            str(row["test_tasks"]),
            str(row["test_entities"]),
            str(row["test_cashtags"]),
            f"{cash_pct:.1f}%",
        )
    table.add_section()
    overall_cash_pct = totals["cash"] / totals["ents"] * 100 if totals["ents"] else 0.0
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{totals['src']}[/bold]",
        f"[bold]{totals['tasks']}[/bold]",
        f"[bold]{totals['ents']}[/bold]",
        f"[bold]{totals['cash']}[/bold]",
        f"[bold]{overall_cash_pct:.1f}%[/bold]",
    )
    console.print(table)


def run(
    seed=DEFAULT_SEED,
    test_fraction=DEFAULT_TEST_FRACTION,
    labeled_folder=LABELED_FOLDER,
    test_folder=TEST_FOLDER,
    quiet=False,
):
    """Programmatic entry point. Same seed → same split, every time.

    Intended to be called from `train.py` at startup so the held-out test
    set always reflects the current labeled data state.
    """
    restore_legacy_test_files(labeled_folder, test_folder)
    clear_existing_test_files(test_folder)
    summary = stratified_split(labeled_folder, test_folder, test_fraction, seed)
    if not summary:
        raise RuntimeError(f"No labeled files found to split in {labeled_folder}.")
    if not quiet:
        _render_summary(summary, seed, test_fraction)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument(
        "--labeled-folder",
        default=LABELED_FOLDER,
        help="Source folder for labeled tasks.",
    )
    parser.add_argument(
        "--test-folder",
        default=TEST_FOLDER,
        help="Output folder for stratified test files.",
    )
    args = parser.parse_args()

    try:
        run(
            seed=args.seed,
            test_fraction=args.test_fraction,
            labeled_folder=args.labeled_folder,
            test_folder=args.test_folder,
        )
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1)

    console.print(
        "\n[green]Next:[/green] run `python trainer/benchmark.py` to rescore all "
        "adapters against the new test set."
    )


if __name__ == "__main__":
    main()

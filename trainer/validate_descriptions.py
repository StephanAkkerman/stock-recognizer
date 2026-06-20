"""Compare entity-description variants on the held-out test set without retraining.

GLiNER2 uses entity descriptions at inference time, so we can measure whether
the updated descriptions reduce FPs before committing to a full training run.
The model is loaded once; inference runs twice (old vs new descriptions).

Usage:
    python trainer/validate_descriptions.py                   # latest adapter, old vs new
    python trainer/validate_descriptions.py --adapter v15
    python trainer/validate_descriptions.py --adapter base    # base model (no adapter)
    python trainer/validate_descriptions.py --threshold 0.6
    python trainer/validate_descriptions.py --show-deltas     # FP/FN diff between variants
    python trainer/validate_descriptions.py --show-deltas --top 30
"""

import argparse
import copy
from collections import Counter

import torch
from rich.console import Console
from rich.table import Table

try:
    from trainer.benchmark import (
        DEFAULT_TEST_FOLDER,
        load_base_model,
        parse_all_label_studio_exports,
        prepare_eval_inputs,
    )
    from trainer.error_analysis import (
        collect_pred_per_doc,
        resolve_adapter,
        run_inference,
    )
except ImportError:
    from benchmark import (
        DEFAULT_TEST_FOLDER,
        load_base_model,
        parse_all_label_studio_exports,
        prepare_eval_inputs,
    )
    from error_analysis import (
        collect_pred_per_doc,
        resolve_adapter,
        run_inference,
    )

console = Console()

# Descriptions used in v15 and earlier — frozen here so the comparison is stable
# even after train.py's ENTITY_DESCRIPTIONS is updated.
OLD_DESCRIPTIONS = {
    "ticker": (
        "A stock market ticker symbol, usually 1-5 letters, often preceded by a dollar sign "
        "(e.g., $AAPL, TSLA). MUST NOT be option strikes, prices, index names, or internet "
        "slang acronyms."
    ),
    "company": (
        "The name of a corporation, hedge fund, or business entity. MUST NOT be an uppercase "
        "ticker symbol, an index, or generic finance terms."
    ),
}

# Updated for v16 — explicitly calls out the FP categories surfaced by error_analysis
NEW_DESCRIPTIONS = {
    "ticker": (
        "A stock market ticker symbol, usually 1-5 letters, often preceded by a dollar sign "
        "(e.g., $AAPL, TSLA). MUST NOT be option expiry months (JAN, FEB, MAR, APR, MAY, JUN, "
        "JUL, AUG, SEP, OCT, NOV, DEC), Reddit slang acronyms (DYOR, NFA, DD, YOLO, FOMO, "
        "TLDR, IMO, AMA, NGL, AH), or financial regulatory bodies (SEC, FINRA, DTC, CSRC, RSA)."
    ),
    "company": (
        "The name of a corporation, hedge fund, or business entity. MUST NOT be an uppercase "
        "ticker symbol, an index, or generic finance terms."
    ),
}


def _compute_metrics(pred_per_doc, gold_per_doc, label_keys):
    counts = {k: {"tp": 0, "fp": 0, "fn": 0} for k in label_keys + ["overall"]}
    for pred, gold in zip(pred_per_doc, gold_per_doc):
        counts["overall"]["tp"] += len(pred & gold)
        counts["overall"]["fp"] += len(pred - gold)
        counts["overall"]["fn"] += len(gold - pred)
        for label in label_keys:
            gp = {e for e in gold if e[1] == label}
            pp = {e for e in pred if e[1] == label}
            counts[label]["tp"] += len(pp & gp)
            counts[label]["fp"] += len(pp - gp)
            counts[label]["fn"] += len(gp - pp)
    metrics = {}
    for key, c in counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        metrics[key] = {"tp": tp, "fp": fp, "fn": fn, "p": p, "r": r, "f1": f1}
    return metrics


def _delta(old_val, new_val, higher_is_better=True, fmt=".2%"):
    diff = new_val - old_val
    if abs(diff) < (0.5 if fmt == "d" else 1e-5):
        return "[dim]—[/dim]"
    sign = "+" if diff > 0 else ""
    color = "green" if (diff > 0) == higher_is_better else "red"
    return f"[{color}]{sign}{diff:{fmt}}[/{color}]"


def _render_comparison(old_m, new_m, label_keys):
    table = Table(title="Entity description comparison (same model, different prompts)", show_lines=False)
    table.add_column("entity", style="blue", width=10)
    table.add_column("metric", width=6)
    table.add_column("old", justify="right")
    table.add_column("new", justify="right")
    table.add_column("Δ", justify="right", width=10)

    rows = [
        ("P (precision)",  "p",  True,  ".2%"),
        ("R (recall)",     "r",  True,  ".2%"),
        ("F1",             "f1", True,  ".2%"),
        ("FP count",       "fp", False, "d"),
        ("FN count",       "fn", False, "d"),
    ]

    for key in label_keys + ["overall"]:
        o = old_m[key]
        n = new_m[key]
        bold = key == "overall"
        label_cell = f"[bold white]{key}[/bold white]" if bold else key
        first = True
        for metric_label, field, hib, fmt in rows:
            table.add_row(
                label_cell if first else "",
                metric_label,
                f"{o[field]:{fmt}}",
                f"{n[field]:{fmt}}",
                _delta(o[field], n[field], higher_is_better=hib, fmt=fmt),
            )
            first = False
        table.add_section()

    return table


def _collect_delta_records(old_preds, new_preds, gold_per_doc, dataset, context_chars=40):
    """Classify spans whose prediction state changed between old and new descriptions.

    Returns four lists of dicts (text, label, doc_idx, context):
      suppressed_fp — old hallucinated it, new correctly skipped it
      new_fp        — new introduced a regression FP
      recovered_fn  — old missed it, new correctly found it
      lost_tp       — old correctly found it, new regressed to miss
    """
    suppressed_fp, new_fp_list, recovered_fn, lost_tp = [], [], [], []

    for doc_idx, (old_pred, new_pred, gold) in enumerate(
        zip(old_preds, new_preds, gold_per_doc)
    ):
        text = dataset[doc_idx]["text"]

        def ctx(s, e):
            l = max(0, s - context_chars)
            r = min(len(text), e + context_chars)
            prefix = "..." if l > 0 else ""
            suffix = "..." if r < len(text) else ""
            snip = (text[l:s] + "[" + text[s:e] + "]" + text[e:r]).replace("\n", " ")
            return f"{prefix}{snip}{suffix}"

        old_fp = old_pred - gold
        new_fp_set = new_pred - gold
        old_fn = gold - old_pred
        new_fn = gold - new_pred

        for span in old_fp - new_fp_set:
            suppressed_fp.append({"text": text[span[0]:span[1]], "label": span[2],
                                   "doc_idx": doc_idx, "context": ctx(span[0], span[1])})
        for span in new_fp_set - old_fp:
            new_fp_list.append({"text": text[span[0]:span[1]], "label": span[2],
                                 "doc_idx": doc_idx, "context": ctx(span[0], span[1])})
        for span in old_fn - new_fn:
            recovered_fn.append({"text": text[span[0]:span[1]], "label": span[2],
                                  "doc_idx": doc_idx, "context": ctx(span[0], span[1])})
        for span in new_fn - old_fn:
            lost_tp.append({"text": text[span[0]:span[1]], "label": span[2],
                             "doc_idx": doc_idx, "context": ctx(span[0], span[1])})

    return suppressed_fp, new_fp_list, recovered_fn, lost_tp


def _render_delta_table(title, records, top):
    table = Table(title=title, show_lines=False)
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("text", style="bold")
    table.add_column("label")
    table.add_column("example context")
    counter = Counter()
    examples = {}
    for r in records:
        key = (r["text"], r["label"])
        counter[key] += 1
        examples.setdefault(key, r)
    for (text, label), count in counter.most_common(top):
        table.add_row(str(count), text, label, examples[(text, label)]["context"])
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="latest",
                        help="'latest' (default), 'base', or version like 'v15'.")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-folder", default=DEFAULT_TEST_FOLDER)
    parser.add_argument("--show-deltas", action="store_true",
                        help="Print per-entity tables of FP/FN changes between variants.")
    parser.add_argument("--top", type=int, default=20,
                        help="Max rows per delta table (default 20).")
    args = parser.parse_args()

    adapter_name, adapter_path = resolve_adapter(args.adapter)
    dataset = parse_all_label_studio_exports(args.test_folder)
    if not dataset:
        console.print(f"[red]No annotated tasks found in {args.test_folder}[/red]")
        raise SystemExit(1)

    label_keys = list(OLD_DESCRIPTIONS.keys())
    flat_chunks, doc_chunk_ranges, gold_per_doc, _ = prepare_eval_inputs(dataset, label_keys)
    total_gold = sum(len(g) for g in gold_per_doc)
    console.print(
        f"Adapter : [bold cyan]{adapter_name}[/bold cyan]\n"
        f"Test set: {len(dataset)} docs | {total_gold} gold entities | "
        f"{len(flat_chunks)} chunks | threshold={args.threshold}\n"
    )

    base_model, device = load_base_model()
    if adapter_path:
        model = copy.deepcopy(base_model)
        model.load_adapter(adapter_path)
        del base_model
    else:
        model = base_model

    run_results = {}
    for variant, descriptions in [("old", OLD_DESCRIPTIONS), ("new", NEW_DESCRIPTIONS)]:
        console.print(f"[cyan]Inference with [bold]{variant}[/bold] descriptions...[/cyan]")
        outputs = run_inference(
            model, flat_chunks, descriptions, args.threshold, args.batch_size
        )
        preds = collect_pred_per_doc(outputs, flat_chunks, doc_chunk_ranges)
        run_results[variant] = {
            "preds": preds,
            "metrics": _compute_metrics(preds, gold_per_doc, label_keys),
        }

    console.print()
    console.print(_render_comparison(
        run_results["old"]["metrics"],
        run_results["new"]["metrics"],
        label_keys,
    ))

    if args.show_deltas:
        suppressed_fp, new_fp_list, recovered_fn, lost_tp = _collect_delta_records(
            run_results["old"]["preds"],
            run_results["new"]["preds"],
            gold_per_doc,
            dataset,
        )
        any_delta = suppressed_fp or new_fp_list or recovered_fn or lost_tp
        if not any_delta:
            console.print(
                "[dim]No prediction differences between old and new descriptions on this test set.[/dim]"
            )
        if suppressed_fp:
            console.print(_render_delta_table(
                f"Suppressed FPs — old hallucinated, new correctly skipped "
                f"({len(suppressed_fp)} instances, {len(set((r['text'], r['label']) for r in suppressed_fp))} unique)",
                suppressed_fp, args.top,
            ))
        if new_fp_list:
            console.print(_render_delta_table(
                f"New FPs — regressions introduced by new descriptions "
                f"({len(new_fp_list)} instances)",
                new_fp_list, args.top,
            ))
        if recovered_fn:
            console.print(_render_delta_table(
                f"Recovered FNs — new descriptions found previously-missed entities "
                f"({len(recovered_fn)} instances)",
                recovered_fn, args.top,
            ))
        if lost_tp:
            console.print(_render_delta_table(
                f"Lost TPs — regressions: old found correctly, new now misses "
                f"({len(lost_tp)} instances)",
                lost_tp, args.top,
            ))

    if device == "cuda":
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

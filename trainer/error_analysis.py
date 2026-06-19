"""Break down where a trained adapter actually fails on the held-out test set.

Run:
    python trainer/error_analysis.py                       # latest adapter
    python trainer/error_analysis.py --adapter v4
    python trainer/error_analysis.py --adapter base
    python trainer/error_analysis.py --threshold 0.5 --top 30
    python trainer/error_analysis.py --save-json out.json  # full dump

Four error categories are surfaced, aggregated by frequency:

  1. **False positives** — predicted entity not in gold, no overlap with a gold
     span. Surface here means "the model hallucinated this token as a ticker /
     company." Highest-leverage signal for hard-negative mining.
  2. **False negatives** — gold entity the model didn't predict at all. Tells
     you what vocabulary or context patterns the model is missing.
  3. **Boundary mismatches** — predicted span overlaps a same-label gold span
     but the offsets differ (e.g. "$AAPL" vs "AAPL", "Apple Inc" vs "Apple").
     These get counted as both FP and FN by the overall metric, exaggerating
     the error.
  4. **Label confusions** — same exact span, different label (e.g. "Apple"
     predicted as ticker but gold says company). Indicates the descriptions
     for ticker/company aren't separating the two cases.

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
        get_all_adapters,
        load_base_model,
        parse_all_label_studio_exports,
        prepare_eval_inputs,
    )
except ImportError:
    from benchmark import (
        DEFAULT_LABELS,
        DEFAULT_TEST_FOLDER,
        get_all_adapters,
        load_base_model,
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


def collect_pred_per_doc(all_outputs, flat_chunks, doc_chunk_ranges):
    """Map chunk-local model outputs back to a per-document set of (start,end,label)."""
    pred_per_doc = []
    for doc_idx, (start, end) in enumerate(doc_chunk_ranges):
        pred = set()
        for chunk_idx in range(start, end):
            _, _, offset = flat_chunks[chunk_idx]
            raw = all_outputs[chunk_idx]
            if isinstance(raw, dict) and "entities" in raw:
                for label, items in raw["entities"].items():
                    for item in items:
                        pred.add((offset + item["start"], offset + item["end"], label))
            elif isinstance(raw, list):
                for item in raw:
                    pred.add((offset + item["start"], offset + item["end"], item["label"]))
        pred_per_doc.append(pred)
    return pred_per_doc


def categorize_errors(pred_per_doc, gold_per_doc, dataset, context_chars=40):
    """Split errors into five categories.

    Each list contains dicts ready for tabular display. Consuming a match
    removes both the FP and FN from the pool before the next pass.

    Pass order (most-specific first):
      1. Label confusions    — exact same span, different label
      2. Boundary mismatches — overlapping spans, same label
      3. Cross-label boundary— overlapping spans, different label
                               (e.g. gold "$FUTU/ticker" vs pred "FUTU/company")
      4. Pure FP / FN        — everything else
    """
    pure_fp = []
    pure_fn = []
    boundary = []
    confusion = []
    cross_boundary = []
    per_doc_counts = []

    for doc_idx, (pred, gold) in enumerate(zip(pred_per_doc, gold_per_doc)):
        text = dataset[doc_idx]["text"]
        fps = pred - gold
        fns = gold - pred

        # 1. Label confusions: exact span match, different label
        fp_by_span = defaultdict(list)
        fn_by_span = defaultdict(list)
        for fp in fps:
            fp_by_span[(fp[0], fp[1])].append(fp)
        for fn in fns:
            fn_by_span[(fn[0], fn[1])].append(fn)

        confused_spans = set(fp_by_span) & set(fn_by_span)
        consumed_fp = set()
        consumed_fn = set()
        for span in confused_spans:
            for fp in fp_by_span[span]:
                for fn in fn_by_span[span]:
                    if fp[2] != fn[2]:
                        confusion.append({
                            "doc_idx": doc_idx,
                            "text": text[span[0]:span[1]],
                            "gold_label": fn[2],
                            "pred_label": fp[2],
                            "context": _make_context(text, span[0], span[1], context_chars),
                        })
                        consumed_fp.add(fp)
                        consumed_fn.add(fn)

        # 2. Boundary mismatches: same label, overlapping spans
        remaining_fps = [fp for fp in fps if fp not in consumed_fp]
        remaining_fns = [fn for fn in fns if fn not in consumed_fn]
        for fp in remaining_fps:
            fs, fe, fl = fp
            best_pair = None
            for fn in remaining_fns:
                ns, ne, nl = fn
                if nl != fl:
                    continue
                if fn in consumed_fn:
                    continue
                if fs < ne and ns < fe:
                    best_pair = fn
                    break
            if best_pair is not None:
                ns, ne, _ = best_pair
                boundary.append({
                    "doc_idx": doc_idx,
                    "gold_text": text[ns:ne],
                    "pred_text": text[fs:fe],
                    "label": fl,
                    "context": _make_context(text, min(fs, ns), max(fe, ne), context_chars),
                })
                consumed_fp.add(fp)
                consumed_fn.add(best_pair)

        # 3. Cross-label boundary: overlapping spans, different label
        # e.g. gold "$FUTU/ticker" vs pred "FUTU/company" — both span and label are off
        for fp in remaining_fps:
            if fp in consumed_fp:
                continue
            fs, fe, fl = fp
            for fn in remaining_fns:
                if fn in consumed_fn:
                    continue
                ns, ne, nl = fn
                if nl == fl:
                    continue  # same-label already handled above
                if fs < ne and ns < fe:  # spans overlap
                    cross_boundary.append({
                        "doc_idx": doc_idx,
                        "gold_text": text[ns:ne],
                        "gold_label": nl,
                        "pred_text": text[fs:fe],
                        "pred_label": fl,
                        "context": _make_context(text, min(fs, ns), max(fe, ne), context_chars),
                    })
                    consumed_fp.add(fp)
                    consumed_fn.add(fn)
                    break

        # 4. Anything still unconsumed is a pure FP / FN
        for fp in fps:
            if fp in consumed_fp:
                continue
            pure_fp.append({
                "doc_idx": doc_idx,
                "text": text[fp[0]:fp[1]],
                "label": fp[2],
                "context": _make_context(text, fp[0], fp[1], context_chars),
            })
        for fn in fns:
            if fn in consumed_fn:
                continue
            pure_fn.append({
                "doc_idx": doc_idx,
                "text": text[fn[0]:fn[1]],
                "label": fn[2],
                "context": _make_context(text, fn[0], fn[1], context_chars),
            })

        per_doc_counts.append({
            "doc_idx": doc_idx,
            "n_errors": len(fps) + len(fns),
            "n_fp": len(fps),
            "n_fn": len(fns),
            "preview": text[:80].replace("\n", " "),
        })

    return {
        "pure_fp": pure_fp,
        "pure_fn": pure_fn,
        "boundary": boundary,
        "cross_boundary": cross_boundary,
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
    return [(key, count, examples[key]) for key, count in counter.most_common(top)]


def _summary_counts(categories, n_tp, total_pred, total_gold):
    """Header row showing the basic TP/FP/FN tally with derived P/R/F1."""
    overlap_errors = (
        len(categories["boundary"])
        + len(categories["cross_boundary"])
        + len(categories["confusion"])
    )
    n_fp = len(categories["pure_fp"]) + overlap_errors
    n_fn = len(categories["pure_fn"]) + overlap_errors
    p = n_tp / total_pred if total_pred else 0.0
    r = n_tp / total_gold if total_gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return n_fp, n_fn, p, r, f1


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


def render_boundary_table(agg):
    table = Table(title="Boundary mismatches (FP overlapping FN, same label)", show_lines=False)
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("gold", style="green")
    table.add_column("predicted", style="yellow")
    table.add_column("label")
    table.add_column("example context")
    for key, count, ex in agg:
        gold_text, pred_text, label = key
        table.add_row(str(count), gold_text, pred_text, label, ex["context"])
    return table


def render_cross_boundary_table(agg):
    table = Table(
        title="Cross-label boundary (overlapping spans, different label — e.g. $FUTU/ticker vs FUTU/company)",
        show_lines=False,
    )
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("gold", style="green")
    table.add_column("gold label", style="green")
    table.add_column("predicted", style="yellow")
    table.add_column("pred label", style="yellow")
    table.add_column("example context")
    for key, count, ex in agg:
        gold_text, gold_label, pred_text, pred_label = key
        table.add_row(str(count), gold_text, gold_label, pred_text, pred_label, ex["context"])
    return table


def render_confusion_table(agg):
    table = Table(title="Label confusions (same span, different label)", show_lines=False)
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("text", style="bold")
    table.add_column("gold→pred")
    table.add_column("example context")
    for key, count, ex in agg:
        text, gold_label, pred_label = key
        table.add_row(str(count), text, f"{gold_label}→{pred_label}", ex["context"])
    return table


def render_doc_hotspots(per_doc, top):
    table = Table(title=f"Top {top} documents by error count", show_lines=False)
    table.add_column("doc", justify="right", style="dim", width=5)
    table.add_column("errors", justify="right")
    table.add_column("FP", justify="right", style="yellow")
    table.add_column("FN", justify="right", style="red")
    table.add_column("preview")
    sorted_docs = sorted(per_doc, key=lambda d: d["n_errors"], reverse=True)[:top]
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
    parser.add_argument("--top", type=int, default=20,
                        help="How many top entries per error category (default 20).")
    parser.add_argument("--context", type=int, default=40,
                        help="Chars of context on each side of an error (default 40).")
    parser.add_argument("--test-folder", default=DEFAULT_TEST_FOLDER,
                        help=f"Held-out test set folder (default {DEFAULT_TEST_FOLDER}).")
    parser.add_argument("--save-json", default=None,
                        help="Optional path to dump the full categorised error data as JSON.")
    args = parser.parse_args()

    adapter_name, adapter_path = resolve_adapter(args.adapter)
    dataset = parse_all_label_studio_exports(args.test_folder)
    if not dataset:
        console.print(f"[red]No data in {args.test_folder}.[/red]")
        raise SystemExit(1)

    label_keys = list(DEFAULT_LABELS.keys())
    flat_chunks, doc_chunk_ranges, gold_per_doc, _ = prepare_eval_inputs(dataset, label_keys)
    total_gold = sum(len(g) for g in gold_per_doc)
    console.print(
        f"Adapter: [bold cyan]{adapter_name}[/bold cyan] @ threshold={args.threshold}\n"
        f"Test set: [bold green]{len(dataset)}[/bold green] docs, "
        f"[bold green]{total_gold}[/bold green] gold entities, "
        f"[bold green]{len(flat_chunks)}[/bold green] chunks"
    )

    base_model, device = load_base_model()
    if adapter_path:
        model = copy.deepcopy(base_model)
        model.load_adapter(adapter_path)
    else:
        model = base_model

    all_outputs = run_inference(model, flat_chunks, DEFAULT_LABELS, args.threshold)
    pred_per_doc = collect_pred_per_doc(all_outputs, flat_chunks, doc_chunk_ranges)
    total_pred = sum(len(p) for p in pred_per_doc)
    n_tp = sum(len(p & g) for p, g in zip(pred_per_doc, gold_per_doc))

    categories = categorize_errors(pred_per_doc, gold_per_doc, dataset, args.context)
    n_fp, n_fn, p, r, f1 = _summary_counts(categories, n_tp, total_pred, total_gold)

    console.print(
        f"\n[bold]TP={n_tp}  FP={n_fp}  FN={n_fn}[/bold]   "
        f"P={p:.2%}  R={r:.2%}  F1={f1:.2%}"
    )
    console.print(
        f"  pure FP: {len(categories['pure_fp'])}  |  "
        f"pure FN: {len(categories['pure_fn'])}  |  "
        f"boundary: {len(categories['boundary'])}  |  "
        f"cross-boundary: {len(categories['cross_boundary'])}  |  "
        f"label confusion: {len(categories['confusion'])}\n"
    )

    if categories["pure_fp"]:
        agg = _aggregate(categories["pure_fp"], ("text", "label"), args.top)
        console.print(render_fp_fn_table(
            f"Top {args.top} false positives (hallucinated entities)", agg, "label"))
    if categories["pure_fn"]:
        agg = _aggregate(categories["pure_fn"], ("text", "label"), args.top)
        console.print(render_fp_fn_table(
            f"Top {args.top} false negatives (missed entities)", agg, "label"))
    if categories["boundary"]:
        agg = _aggregate(categories["boundary"], ("gold_text", "pred_text", "label"), args.top)
        console.print(render_boundary_table(agg))
    if categories["cross_boundary"]:
        agg = _aggregate(
            categories["cross_boundary"],
            ("gold_text", "gold_label", "pred_text", "pred_label"),
            args.top,
        )
        console.print(render_cross_boundary_table(agg))
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

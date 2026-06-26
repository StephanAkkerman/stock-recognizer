"""Patch annotation gaps in data/labeled/ from an engine-mode error JSON.

Targets a fixed set of tickers that the engine correctly finds but the gold
annotations missed. Reads errors_full.json (produced by
``error_analysis.py --engine --test-folder data/labeled --save-json``),
locates each span in the source document, and writes the new annotation back
into the Label Studio JSON files.

Usage::

    # Dry run — show what would be patched
    python trainer/patch_labeled_gaps.py

    # Apply patches
    python trainer/patch_labeled_gaps.py --apply
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys

try:
    from trainer.patch_test_labels import find_entity_span, make_ls_result
except ImportError:
    from patch_test_labels import find_entity_span, make_ls_result

# Tickers confirmed to be annotation gaps (engine is correct, gold missed them).
TARGET_TICKERS = {"GOOGL", "UBER", "BYND", "SPOT", "RDDT"}

ERRORS_FILE = "errors_full.json"
LABELED_FOLDER = "data/labeled"


def load_labeled_docs_with_provenance(folder: str) -> list[dict]:
    """Load all labeled docs in the same glob order as parse_all_label_studio_exports.

    Returns a list where index == doc_idx used by the error analysis run that
    produced errors_full.json (which used --test-folder data/labeled).
    """
    docs = []
    for file_path in glob.glob(f"{folder}/*.json"):
        with open(file_path, encoding="utf-8") as f:
            tasks = json.load(f)
        for task_idx, task in enumerate(tasks):
            if not task.get("annotations") or task["annotations"][0].get("was_cancelled"):
                continue
            docs.append({
                "text": task["data"]["text"],
                "_file": file_path,
                "_task_idx": task_idx,
                "_task_id": task.get("id"),
            })
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--errors", default=ERRORS_FILE)
    parser.add_argument("--labeled-folder", default=LABELED_FOLDER)
    parser.add_argument("--apply", action="store_true",
                        help="Write patches to the Label Studio JSON files.")
    args = parser.parse_args()

    with open(args.errors, encoding="utf-8") as f:
        error_data = json.load(f)

    if error_data.get("mode") != "engine":
        sys.exit(f"Expected engine-mode error JSON, got mode={error_data.get('mode')!r}")

    target_fps = [r for r in error_data["fp"] if r["text"] in TARGET_TICKERS]
    print(f"Found {len(target_fps)} FP records for target tickers: "
          f"{sorted({r['text'] for r in target_fps})}")

    docs = load_labeled_docs_with_provenance(args.labeled_folder)
    print(f"Loaded {len(docs)} labeled documents from {args.labeled_folder}")

    patches_by_file: dict[str, list[dict]] = {}
    skipped = []

    for record in target_fps:
        doc_idx = record["doc_idx"]
        ticker = record["text"]
        context = record["context"]

        if doc_idx >= len(docs):
            print(f"  WARNING: doc_idx={doc_idx} out of range")
            skipped.append(record)
            continue

        doc = docs[doc_idx]
        doc_text = doc["text"]

        # Try bare ticker first, then $TICKER (cashtag form).
        # _find_ticker_context highlights whichever form appears in the doc,
        # so the context may contain [GOOGL] or [$GOOGL] depending on the text.
        span = find_entity_span(doc_text, ticker, context)
        if span is None:
            span = find_entity_span(doc_text, f"${ticker}", context)

        if span is None:
            print(f"  WARNING: could not locate [{ticker}] span in doc {doc_idx} — skipped")
            skipped.append(record)
            continue

        start, end = span
        surface = doc_text[start:end]
        print(f"  doc {doc_idx:4d}  {ticker}  [{start}:{end}]  {surface!r:12s}  "
              f"{context[:70]}")

        entry = {
            "task_idx": doc["_task_idx"],
            "start": start,
            "end": end,
            "label": "ticker",
            "text": surface,
        }
        patches_by_file.setdefault(doc["_file"], []).append(entry)

    total = sum(len(v) for v in patches_by_file.values())
    print(f"\n{total} span(s) located across {len(patches_by_file)} file(s).")
    if skipped:
        print(f"{len(skipped)} record(s) skipped (span not found).")

    if not args.apply:
        print("\n[Dry run] Pass --apply to write patches.")
        return

    for file_path, patches in patches_by_file.items():
        with open(file_path, encoding="utf-8") as f:
            tasks = json.load(f)

        added = 0
        for patch in patches:
            task = tasks[patch["task_idx"]]
            result = task["annotations"][0].setdefault("result", [])
            already = any(
                r.get("type") == "labels"
                and r["value"]["start"] == patch["start"]
                and r["value"]["end"] == patch["end"]
                and "ticker" in r["value"].get("labels", [])
                for r in result
            )
            if not already:
                result.append(make_ls_result(patch["start"], patch["end"],
                                             patch["label"], patch["text"]))
                added += 1

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)
        print(f"  {added} addition(s) → {file_path}")

    print("Done. Re-run the benchmark to verify metrics.")


if __name__ == "__main__":
    main()

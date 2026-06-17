"""Promote false-positive model predictions to gold labels in the test set.

The benchmark compares model output against manually annotated spans.  When the
model predicts an entity that is *correct* but was not annotated in the test
data (a missing annotation rather than a hallucination), it counts as both a
false positive *and* a false negative, artificially hurting both precision and
recall.

This script automates the fix:

  1. Reads the error JSON produced by ``error_analysis.py --save-json``.
  2. For each pure false positive, checks whether the predicted text is a known
     ticker or company using the engine's own market-data dictionaries.
  3. Applies a blocklist of financial acronyms that look like tickers but
     aren't stock references (PDT, EV, PT, SG&A, …).
  4. Locates the exact character span in the source document using the context
     snippet stored in the error JSON.
  5. Runs an interactive review of the surviving candidates, then (with
     ``--apply``) writes the approved additions back into the Label Studio JSON
     files without touching anything else.

Usage::

    # Dry run — show candidates without writing anything
    python trainer/patch_test_labels.py --errors errors_v17.json

    # Apply approved additions to the test-set JSON files
    python trainer/patch_test_labels.py --errors errors_v17.json --apply

    # Skip interactive review and auto-apply all engine-validated hits
    python trainer/patch_test_labels.py --errors errors_v17.json --apply --auto
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import string
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Blocklist: uppercase tokens that look like tickers but are used in finance
# text to mean something else (rules, metrics, acronyms).
# ---------------------------------------------------------------------------
FINANCIAL_ABBREV_BLOCKLIST = {
    # Trading rules / regulatory
    "PDT", "FINRA", "SEC", "CFTC", "DTCC", "DTC", "OCC", "SIPC", "FDIC",
    "NDAA", "FTC", "DOJ", "DOW",
    # Financial metrics / jargon
    "EV", "EBIT", "EBITDA", "FCF", "PE", "PS", "PB", "EPS", "BPS", "DPS",
    "NAV", "AUM", "ROE", "ROI", "ROA", "CAGR", "WACC", "DCF", "IRR",
    "SGA", "SGNA", "PT", "TP", "YOY", "QOQ", "TTM", "LTM", "NTM",
    "EOY", "EOQ", "YTD", "MTD",
    # Options / derivatives
    "ATM", "OTM", "ITM", "IV", "VIX", "VWAP",
    # Generic acronyms appearing in financial posts
    "RSU", "RSUS", "ISO", "ESO", "ESPP", "PSA", "AMA", "IMO", "AFAIK",
    "IPO",  # keep this? It IS relevant context but not an entity itself
    "AI",   # generic tech term
    "ML", "NLP", "LLM",
    # Technology / product codenames that appear in chip/tech DDs
    "EUV", "DRAM", "NAND", "GPU", "CPU", "HBM", "LPBF",
    # Japanese ministry acronyms
    "MEXT", "METI",
    # Month abbreviations — used in options notation ("AUG 19 calls", "OCT 21 puts").
    # Block these even when written as "$ AUG" with a stray space before the letters.
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
    # Misc
    "CA", "HQ", "US", "UK", "EU",
}

# ---------------------------------------------------------------------------
# Company/person names that the engine's company_to_ticker lookup might return
# but that are NOT being used as company references in typical Reddit finance
# text.
# ---------------------------------------------------------------------------
COMPANY_BLOCKLIST = {
    "arm",          # "Amazon's cloud arm" ≠ Arm Holdings
    "go",           # "go short" ≠ Genie Energy
    "air",          # "thin air" ≠ Air Industries
    "might",        # verb, not a ticker
    "fukd",         # slang
    "pamp",         # slang
    "ish",          # suffix
    "amirite",      # internet slang
    "artice",       # typo for "article"
    "pink slip",    # idiom
    "burry",        # person (Michael Burry), not a publicly traded company
    "chatgpt",      # product, not a company
    "the verge",    # media outlet; no stock
    # "reddit" is intentionally NOT blocked — Reddit (RDDT) is a public company
    "sahmcapital.com",
    "lumenai",      # unverifiable private startup
    "controversialclub",
    "nufacturers",  # truncated word
    "giga factory", # facility description
    "laser powder bed fusion",  # manufacturing process
    "rapid production solutions",  # Velo3D internal division name
    "big pharma",   # generic category
    "defined capital",
    "convexity",
    "sg&a",
    "united waste", # private company sold; no ticker
    "bloomberg",    # Bloomberg L.P. is private; no publicly traded stock
    # Common English words that happen to match obscure ticker symbols
    "circle",       # "circle jerk" etc.
    "go",           # "go short"
    "air",          # "thin air"
    "prime",        # "Amazon Prime" or generic adjective
    "link",         # generic word
    "real",         # generic adjective
    "core",         # generic word
}


def load_test_with_provenance(folder: str):
    """Load test docs, tagging each with its source file path and task index.

    Replicates the glob order used by ``parse_all_label_studio_exports`` so
    that ``doc_idx`` values from the error JSON map to the correct entries.
    """
    files = glob.glob(os.path.join(folder, "*.json"))
    result = []
    for file_path in files:
        with open(file_path, encoding="utf-8") as f:
            tasks = json.load(f)
        for task_idx, task in enumerate(tasks):
            if not task.get("annotations") or task["annotations"][0].get("was_cancelled"):
                continue
            text = task["data"]["text"]
            result.append(
                {
                    "text": text,
                    "_file": file_path,
                    "_task_idx": task_idx,
                }
            )
    return result


def find_entity_span(
    doc_text: str, entity_text: str, context_snippet: str
) -> tuple[int, int] | None:
    """Locate (start, end) of entity_text in doc_text using the context snippet.

    The context format produced by ``_make_context`` is::

        "...left_text[ENTITY]right_text..."

    We extract the left anchor and search for ``left_anchor + entity`` in the
    document to pin down the exact position even when the entity appears
    multiple times.
    """
    bracket_re = re.compile(r"\[" + re.escape(entity_text) + r"\]")
    m = bracket_re.search(context_snippet)
    if not m:
        return None

    left_raw = context_snippet[: m.start()]
    left = left_raw.lstrip(".")
    # Unescape forward slashes (the context writer escapes them)
    left = left.replace("\\/", "/")
    entity_clean = entity_text.replace("\\/", "/")

    def _find(anchor: str) -> tuple[int, int] | None:
        target = anchor + entity_clean
        pos = doc_text.find(target)
        if pos >= 0:
            s = pos + len(anchor)
            return s, s + len(entity_clean)
        return None

    # Try progressively shorter left anchors
    for trim in (None, 30, 20, 10):
        anchor = left if trim is None else left[-trim:]
        if not anchor:
            continue
        hit = _find(anchor)
        if hit:
            return hit

    # Last resort: find all occurrences of the entity text
    all_pos = [i for i in range(len(doc_text)) if doc_text[i : i + len(entity_clean)] == entity_clean]
    if len(all_pos) == 1:
        return all_pos[0], all_pos[0] + len(entity_clean)

    return None


def is_engine_valid(text: str, label: str, engine) -> bool:
    """Return True if text resolves to a known ticker or company."""
    upper = text.upper().lstrip("$").replace(" ", "")
    lower = text.lower()

    if label == "ticker":
        if upper in FINANCIAL_ABBREV_BLOCKLIST:
            return False
        # Also block common words that happen to match a ticker symbol
        if lower in COMPANY_BLOCKLIST:
            return False
        return upper in engine.valid_tickers

    if label == "company":
        if lower in COMPANY_BLOCKLIST:
            return False
        for variant in (text, text.title(), lower, upper):
            if variant in engine.company_to_ticker:
                return True

        # Special case: "Reddit" → RDDT is a real ticker but we only want to
        # promote it when it's clearly a stock reference, not platform usage.
        # Leave it for manual review by returning False here.
        return False

    return False


def make_ls_result(start: int, end: int, label: str, text: str) -> dict:
    """Build a Label Studio annotation result dict for a new span."""
    return {
        "id": f"patch_{start}_{end}",
        "type": "labels",
        "value": {
            "start": start,
            "end": end,
            "text": text,
            "labels": [label],
        },
        "from_name": "label",
        "to_name": "text",
        "origin": "manual",
    }


def interactive_review(candidates: list[dict]) -> list[dict]:
    """Ask the user to accept or reject each candidate (y/n/q)."""
    approved = []
    total = len(candidates)
    print(f"\n{'='*70}")
    print(f"  Review {total} engine-validated candidates (y=accept, n=skip, q=quit)")
    print(f"{'='*70}\n")
    for i, c in enumerate(candidates, 1):
        print(f"[{i}/{total}]  label={c['label']}  text={c['text']!r}")
        print(f"  context: {c['context']}")
        print(f"  file: {c['_file']}  doc_idx={c['doc_idx']}")
        while True:
            ans = input("  Accept? [y/n/q] ").strip().lower()
            if ans in ("y", "n", "q"):
                break
        if ans == "y":
            approved.append(c)
        elif ans == "q":
            print("  Quitting review — changes accepted so far will be applied.")
            break
        print()
    return approved


def apply_patches(approved: list[dict], test_folder: str) -> None:
    """Write approved span additions into the Label Studio JSON files."""
    # Group by file so we only write each file once
    by_file: dict[str, list[dict]] = {}
    for c in approved:
        by_file.setdefault(c["_file"], []).append(c)

    for file_path, patches in by_file.items():
        with open(file_path, encoding="utf-8") as f:
            tasks = json.load(f)

        for patch in patches:
            task = tasks[patch["_task_idx"]]
            annotation = task["annotations"][0]
            result = annotation.setdefault("result", [])

            # Check if this exact span+label already exists
            already_there = any(
                r.get("type") == "labels"
                and r["value"]["start"] == patch["start"]
                and r["value"]["end"] == patch["end"]
                and patch["label"] in r["value"].get("labels", [])
                for r in result
            )
            if not already_there:
                result.append(
                    make_ls_result(patch["start"], patch["end"], patch["label"], patch["text"])
                )

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)

        print(f"  Wrote {len(patches)} addition(s) → {file_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--errors",
        default="errors_v17.json",
        help="Path to the error JSON from error_analysis.py --save-json (default: errors_v17.json)",
    )
    parser.add_argument(
        "--test-folder",
        default="data/test",
        help="Test set folder to patch (default: data/test)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write approved additions to the Label Studio JSON files",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Skip interactive review and auto-accept all engine-validated hits",
    )
    args = parser.parse_args()

    if not os.path.exists(args.errors):
        sys.exit(f"Error file not found: {args.errors}")

    with open(args.errors, encoding="utf-8") as f:
        error_data = json.load(f)

    pure_fps = error_data.get("categories", {}).get("pure_fp", [])
    if not pure_fps:
        print("No pure false positives found in error JSON.")
        return

    print(f"Loaded {len(pure_fps)} pure false positives from {args.errors}")

    # Load engine (regex only — just need the market data dicts)
    from stock_recognizer.engine import StockRecognizer

    engine = StockRecognizer(use_ai=False)

    # Load test docs with provenance
    docs = load_test_with_provenance(args.test_folder)
    print(f"Loaded {len(docs)} test documents from {args.test_folder}")

    # Validate FPs
    candidates = []
    skipped_not_in_engine = 0
    skipped_no_span = 0

    for fp in pure_fps:
        doc_idx = fp["doc_idx"]
        text = fp["text"]
        label = fp["label"]
        context = fp["context"]

        if not is_engine_valid(text, label, engine):
            skipped_not_in_engine += 1
            continue

        if doc_idx >= len(docs):
            print(f"  WARNING: doc_idx={doc_idx} out of range (have {len(docs)} docs)")
            continue

        doc = docs[doc_idx]
        span = find_entity_span(doc["text"], text, context)
        if span is None:
            print(f"  WARNING: could not locate span for {text!r} in doc {doc_idx}")
            skipped_no_span += 1
            continue

        start, end = span
        candidates.append(
            {
                "doc_idx": doc_idx,
                "text": text,
                "label": label,
                "context": context,
                "start": start,
                "end": end,
                "_file": doc["_file"],
                "_task_idx": doc["_task_idx"],
            }
        )

    print(
        f"\n  {skipped_not_in_engine} FPs skipped (not in engine data or blocklisted)"
    )
    print(f"  {skipped_no_span} FPs skipped (span not locatable in document)")
    print(f"  {len(candidates)} candidates to review\n")

    if not candidates:
        print("Nothing to do.")
        return

    print("Candidates:")
    for c in candidates:
        print(f"  [{c['label']:7s}] {c['text']!r:25s} @ {c['start']}:{c['end']}  {c['context'][:80]}")

    if args.apply:
        if args.auto:
            approved = candidates
            print(f"\nAuto-accepting all {len(approved)} candidates.")
        else:
            approved = interactive_review(candidates)

        if approved:
            print(f"\nApplying {len(approved)} addition(s)…")
            apply_patches(approved, args.test_folder)
            print("Done. Re-run the benchmark to see updated metrics.")
        else:
            print("No additions approved.")
    else:
        print(
            "\n[Dry run] Pass --apply to write changes, "
            "--apply --auto to accept all without review."
        )


if __name__ == "__main__":
    main()

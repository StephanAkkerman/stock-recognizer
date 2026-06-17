"""Audit training data annotations for incorrectly labeled spans.

Finds and optionally removes annotations that match known-bad patterns:

  - Financial / regulatory acronyms (PDT, EV, SEC, AUG as a month expiry…)
  - Common words that happen to resolve via company_to_ticker but aren't
    being used as stock references (arm, bloomberg, circle, go…)
  - ``$ X`` cashtag-with-space spans (``$`` followed by a space — not a real
    cashtag; typically options notation like "$ AUG 19 calls")
  - Single-character labeling accidents

This is the training-data complement to ``patch_test_labels.py`` (which
promotes model false positives to test-set ground truth) and sits alongside
``fix_label_policy.py`` (which handles ticker/company relabeling).

Usage::

    python trainer/audit_train_labels.py               # dry run
    python trainer/audit_train_labels.py --apply       # write fixes
    python trainer/audit_train_labels.py --dirs data/labeled data/augmented
    python trainer/audit_train_labels.py --verbose     # show context per hit
"""

import argparse
import glob
import json
import os
from collections import defaultdict

# ---------------------------------------------------------------------------
# Spans (normalized to uppercase, $ stripped, spaces removed) that should
# never be annotated as tickers or companies in Reddit finance posts.
# ---------------------------------------------------------------------------
ABBREV_BLOCKLIST = {
    # Regulatory / government bodies
    "PDT",
    "FINRA",
    "SEC",
    "CFTC",
    "DTCC",
    "DTC",
    "OCC",
    "SIPC",
    "FDIC",
    "NDAA",
    "FTC",
    "DOJ",
    "CSRC",
    "NASA",
    "MEXT",
    "METI",
    # Financial metrics and jargon
    "EV",
    "EBIT",
    "EBITDA",
    "FCF",
    "PE",
    "PS",
    "PB",
    "EPS",
    "BPS",
    "DPS",
    "NAV",
    "AUM",
    "ROE",
    "ROI",
    "ROA",
    "CAGR",
    "WACC",
    "DCF",
    "IRR",
    "SGA",
    "SGNA",
    "PT",
    "TP",
    "YOY",
    "QOQ",
    "TTM",
    "LTM",
    "NTM",
    "EOY",
    "EOQ",
    "YTD",
    "MTD",
    # Options / derivatives jargon
    "ATM",
    "OTM",
    "ITM",
    "IV",
    # VIX intentionally excluded — labeled as ticker in training data (index reference)
    "VWAP",
    # Generic acronyms that appear everywhere in finance posts
    "RSU",
    "RSUS",
    "ISO",
    "ESO",
    "ESPP",
    "PSA",
    "AMA",
    "IMO",
    "AFAIK",
    "IPO",
    "AI",
    "ML",
    "NLP",
    "LLM",
    # Technology product types, not companies
    "EUV",
    "DRAM",
    "NAND",
    "GPU",
    "CPU",
    "HBM",
    "LPBF",
    # Month abbreviations used in options notation ("AUG 19 calls", "OCT 21 puts")
    # — even when written as "$ AUG" with a stray space, they are NOT cashtags
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
    # Media outlets with no public stock
    "CNBC",
    "MSNBC",
    "HBO",
    "BBC",
    "WSJ",
    # Short geographic / legal suffixes misread as tickers
    "CA",
    "HQ",
    "US",
    "UK",
    "EU",
}

# ---------------------------------------------------------------------------
# Phrases / words (lowercased) that resolve via company_to_ticker but are
# almost never being used as stock references in Reddit posts.
# ---------------------------------------------------------------------------
WORD_BLOCKLIST = {
    "arm",  # "Amazon's cloud arm" ≠ Arm Holdings
    "go",  # verb ("go short", "go long")
    "air",  # noun ("thin air")
    "might",  # modal verb
    "circle",  # generic noun ("full circle")
    "prime",  # adjective or "Amazon Prime"
    "link",  # generic noun
    "real",  # adjective
    "core",  # generic noun
    "big pharma",  # generic category, not a company
    "bloomberg",  # Bloomberg L.P. — private, not publicly traded
    "the verge",  # media outlet
    "chatgpt",  # product
    "burry",  # person (Michael Burry)
    "fed",  # institution / verb ("fed up")
    "the fed",  # institution
    "sg&a",  # financial metric phrase
    "united waste",  # sold; no active ticker
    "defined capital",
    "convexity",
    "lumenai",
    "controversialclub",
    "nufacturers",  # truncated word
    "giga factory",
    "laser powder bed fusion",
    "rapid production solutions",
}


def classify_violation(text: str, label: str) -> str | None:
    """Return a violation category string, or None if the span looks OK.

    Categories
    ----------
    space_cashtag    ``$`` followed by a space — not a real cashtag
    single_char      Single-character span — almost always a labeling accident
    abbrev_blocklist Financial/regulatory acronym or month abbreviation
    word_blocklist   Common word falsely matching company_to_ticker
    """
    # "$ X" with a space between $ and the word is NOT a cashtag
    if text.startswith("$") and len(text) > 1 and text[1].isspace():
        return "space_cashtag"

    # Real cashtag ($ directly attached, e.g. $H, $DRAM, $EUV) — trusted
    # unconditionally; matches the engine's pipeline rule.
    if text.startswith("$") and len(text) > 1 and not text[1].isspace():
        return None

    clean = text.strip("$").strip()
    upper = clean.upper().replace(" ", "")
    lower = text.strip().lower()

    if len(upper) <= 1:
        return "single_char"

    if upper in ABBREV_BLOCKLIST:
        return "abbrev_blocklist"

    # WORD_BLOCKLIST targets context-dependent words (common nouns that happen
    # to match company_to_ticker). Only apply to non-all-caps text: an all-caps
    # span like ARM is the ARM Holdings ticker, not the common noun "arm".
    is_all_caps = clean == clean.upper()
    if not is_all_caps and lower in WORD_BLOCKLIST:
        return "word_blocklist"

    return None


def audit_file(file_path: str) -> list[dict]:
    """Return a list of violation dicts found in one Label Studio JSON file."""
    with open(file_path, encoding="utf-8") as f:
        tasks = json.load(f)

    violations = []
    for task_idx, task in enumerate(tasks):
        doc_text = task["data"]["text"]
        for ann in task.get("annotations") or []:
            if ann.get("was_cancelled"):
                continue
            for result_idx, r in enumerate(ann.get("result", [])):
                if r.get("type") != "labels":
                    continue
                v = r["value"]
                text = v.get("text") or doc_text[v["start"] : v["end"]]
                label = v["labels"][0]

                reason = classify_violation(text, label)
                if reason is None:
                    continue

                start = v.get("start", 0)
                end = v.get("end", start + len(text))
                left = doc_text[max(0, start - 40) : start]
                right = doc_text[end : min(len(doc_text), end + 40)]
                context = f"…{left}[{text}]{right}…"

                violations.append(
                    {
                        "file": file_path,
                        "task_idx": task_idx,
                        "result_idx": result_idx,
                        "text": text,
                        "label": label,
                        "reason": reason,
                        "context": context,
                    }
                )

    return violations


def apply_fixes(violations: list[dict]) -> None:
    """Remove flagged annotations from the Label Studio JSON files."""
    by_file: dict[str, list[dict]] = defaultdict(list)
    for v in violations:
        by_file[v["file"]].append(v)

    for file_path, file_violations in by_file.items():
        with open(file_path, encoding="utf-8") as f:
            tasks = json.load(f)

        # Collect result indices to remove, grouped by task
        by_task: dict[int, list[int]] = defaultdict(list)
        for viol in file_violations:
            by_task[viol["task_idx"]].append(viol["result_idx"])

        # Remove in reverse index order so earlier indices stay valid
        for task_idx, indices in by_task.items():
            result = tasks[task_idx]["annotations"][0]["result"]
            for ridx in sorted(set(indices), reverse=True):
                result.pop(ridx)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)

        print(
            f"  {os.path.basename(file_path)}: removed {len(file_violations)} span(s)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=["data/labeled"],
        help="Folders to scan (default: data/labeled)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Remove flagged annotations; omit for dry run",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print context string for every violation",
    )
    args = parser.parse_args()

    all_violations: list[dict] = []
    for folder in args.dirs:
        for fp in sorted(glob.glob(os.path.join(folder, "*.json"))):
            viol = audit_file(fp)
            all_violations.extend(viol)

    if not all_violations:
        print("No violations found.")
        return

    unique_files = len(set(v["file"] for v in all_violations))
    print(
        f"\nFound {len(all_violations)} violation(s) across {unique_files} file(s).\n"
    )

    reason_labels = {
        "space_cashtag": "$ with space (not a cashtag)",
        "single_char": "Single-character span",
        "abbrev_blocklist": "Financial/regulatory acronym or month",
        "word_blocklist": "Common word falsely matching company",
    }

    by_reason: dict[str, list[dict]] = defaultdict(list)
    for v in all_violations:
        by_reason[v["reason"]].append(v)

    for reason in (
        "space_cashtag",
        "single_char",
        "abbrev_blocklist",
        "word_blocklist",
    ):
        vlist = by_reason.get(reason)
        if not vlist:
            continue
        print(f"[{reason_labels[reason]}]  ×{len(vlist)}")
        counts: dict[tuple, int] = defaultdict(int)
        for v in vlist:
            counts[(v["text"], v["label"])] += 1
        for (text, label), n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {text!r:30s}  ({label})  ×{n}")
        if args.verbose:
            for v in vlist:
                print(f"    {v['context'][:100]}")
                print(f"    → {os.path.basename(v['file'])}")
        print()

    if args.apply:
        print("Removing violations…")
        apply_fixes(all_violations)
        print("\nDone. Regenerate data/augmented/ before retraining.")
    else:
        print("[Dry run] Pass --apply to remove flagged annotations.")


if __name__ == "__main__":
    main()

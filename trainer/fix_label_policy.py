"""Audit and fix ticker/company label policy violations in annotated data.

The labeling policy (documented in CLAUDE.md) uses the engine's own market-data
dictionaries as the source of truth:

  1. Span (uppercased, $-stripped) in valid_tickers  →  label must be ``ticker``
  2. Span resolves via company_to_ticker but NOT in valid_tickers  →  ``company``
  3. Neither resolves  →  flag; do not label government bodies / financial acronyms

Run::

    python trainer/fix_label_policy.py                      # dry run
    python trainer/fix_label_policy.py --apply              # write fixes
    python trainer/fix_label_policy.py --dirs data/labeled  # one folder only
"""

import argparse
import glob
import json
import os
from collections import defaultdict

# ---------------------------------------------------------------------------
# Entities that should never be annotated: government bodies, regulatory
# agencies, financial acronyms used as common terms, and media companies
# with no public ticker.
# ---------------------------------------------------------------------------
REMOVE_SET = {
    # Regulatory / government
    "CSRC",
    "SEC",
    "FINRA",
    "CFTC",
    "NASA",
    "NDAA",
    "MEXT",
    "METI",
    "FTC",
    "DOJ",
    "FDIC",
    "SIPC",
    # Financial metric acronyms
    "PDT",
    "EV",
    "EBITDA",
    "EBIT",
    "FCF",
    "SGA",
    "SGNA",
    "PT",
    "TP",
    "EPS",
    "BPS",
    "NAV",
    "AUM",
    "ROE",
    "ROI",
    "DCF",
    "WACC",
    "EOY",
    "YTD",
    "YOY",
    "QOQ",
    "TTM",
    "LTM",
    "RSU",
    "RSUS",
    "ESPP",
    "PSA",
    "ATM",
    "OTM",
    "ITM",
    # Tech acronyms / product types (not companies)
    "EUV",
    "DRAM",
    "NAND",
    "GPU",
    "CPU",
    "HBM",
    "LPBF",
    "AI",
    "ML",
    # Media outlets with no public ticker
    "CNBC",
    "MSNBC",
    "HBO",
    "BBC",
}


def load_engine():
    from stock_recognizer.engine import StockRecognizer

    return StockRecognizer(use_ai=False)


def expected_label(text: str, engine) -> str | None:
    """Return 'ticker', 'company', or None (remove) for a span.

    Form drives the decision — casing in the source text is the signal:

    * Cashtag / ALL-CAPS  → check valid_tickers first (ticker), then company
    * all-lowercase       → check valid_tickers first (informal shorthand), then company
    * Mixed-case          → company only; never auto-relabel to ticker
    """
    from stock_recognizer.constants import AMBIGUOUS_WORDS

    clean = text.strip("$")
    upper = clean.upper().replace(" ", "")
    lower = clean.lower()

    # Skip single-character spans (almost always labeling accidents)
    if len(upper) <= 1:
        return "unknown"

    is_cashtag = text.startswith("$")

    # --- Cashtag ($AAPL, $DRAM, $EUV): trusted unconditionally, matches pipeline ---
    # Must come BEFORE the remove set — a $ prefix overrides word-level blocklists.
    if is_cashtag:
        return "ticker"

    # Explicit remove set: government bodies, financial metric acronyms, media outlets
    if upper in REMOVE_SET:
        return None

    # Skip words already handled by the engine's AMBIGUOUS_WORDS list — too
    # risky to auto-relabel (e.g. OPEN, GAP, GO, USA, HERE)
    if upper in AMBIGUOUS_WORDS:
        return "unknown"

    is_all_caps = clean == clean.upper() and len(clean) >= 2
    is_all_lower = clean == clean.lower() and len(clean) >= 2
    # Anything else is mixed-case (e.g. "Meta", "SoFi", "Velo3D")

    # --- ALL-CAPS: looks like a ticker symbol ---
    if is_all_caps:
        if upper in engine.valid_tickers:
            return "ticker"
        # All-caps but ticker differs (NVIDIA→NVDA, TSMC→TSM): stays company
        for v in (clean, lower, lower.capitalize()):
            if v in engine.company_to_ticker:
                return "company"
        return "unknown"

    # --- all-lowercase: informal ticker shorthand (gme, amc, tsla) ---
    if is_all_lower:
        if upper in engine.valid_tickers:
            return "ticker"
        for v in (clean.capitalize(), clean.title()):
            if v in engine.company_to_ticker:
                return "company"
        return "unknown"

    # --- Mixed-case: cannot reliably distinguish company name from informal ticker ---
    # "Meta" (company name) and "Gsx" (informal GSX ticker) are both mixed-case
    # but need different labels. The engine's data can't tell them apart because
    # company_to_ticker stores only uppercase keys and valid_tickers matches both.
    # Leave mixed-case labels as annotated; fix manually via error_analysis.py.
    return "unknown"


def audit_file(file_path: str, engine) -> list[dict]:
    """Return a list of issues found in one Label Studio JSON file."""
    with open(file_path, encoding="utf-8") as f:
        tasks = json.load(f)

    issues = []
    for task_idx, task in enumerate(tasks):
        for ann in task.get("annotations") or []:
            if ann.get("was_cancelled"):
                continue
            for result_idx, r in enumerate(ann.get("result", [])):
                if r.get("type") != "labels":
                    continue
                v = r["value"]
                text = v.get("text") or task["data"]["text"][v["start"] : v["end"]]
                current = v["labels"][0]
                expected = expected_label(text, engine)

                if expected == "unknown":
                    continue  # can't determine; skip

                if expected is None:
                    issues.append(
                        {
                            "file": file_path,
                            "task_idx": task_idx,
                            "result_idx": result_idx,
                            "text": text,
                            "current": current,
                            "action": "remove",
                        }
                    )
                elif expected != current:
                    issues.append(
                        {
                            "file": file_path,
                            "task_idx": task_idx,
                            "result_idx": result_idx,
                            "text": text,
                            "current": current,
                            "action": f"relabel→{expected}",
                            "expected": expected,
                        }
                    )

    return issues


def apply_fixes(issues: list[dict]) -> None:
    """Write fixes back to the Label Studio JSON files."""
    # Group by file so each file is only written once
    by_file: dict[str, list[dict]] = defaultdict(list)
    for issue in issues:
        by_file[issue["file"]].append(issue)

    for file_path, file_issues in by_file.items():
        with open(file_path, encoding="utf-8") as f:
            tasks = json.load(f)

        # Sort removes last so result_idx stays valid during relabels
        file_issues_sorted = sorted(
            file_issues,
            key=lambda x: (x["task_idx"], x["result_idx"], x["action"] == "remove"),
        )
        # Track removals: process in reverse order per task to avoid index shifting
        removes_by_task: dict[int, list[int]] = defaultdict(list)

        for issue in file_issues_sorted:
            tidx = issue["task_idx"]
            ridx = issue["result_idx"]
            task = tasks[tidx]
            annotation = task["annotations"][0]
            result = annotation["result"]

            if issue["action"] == "remove":
                removes_by_task[tidx].append(ridx)
            else:
                result[ridx]["value"]["labels"] = [issue["expected"]]

        # Apply removes in reverse index order per task
        for tidx, indices in removes_by_task.items():
            result = tasks[tidx]["annotations"][0]["result"]
            for ridx in sorted(indices, reverse=True):
                result.pop(ridx)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)

        relabels = sum(1 for i in file_issues if i["action"] != "remove")
        removes = sum(1 for i in file_issues if i["action"] == "remove")
        print(
            f"  {os.path.basename(file_path)}: {relabels} relabeled, {removes} removed"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=["data/labeled", "data/test"],
        help="Folders to scan (default: data/labeled data/test)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write fixes; omit for dry run",
    )
    args = parser.parse_args()

    engine = load_engine()

    all_issues: list[dict] = []
    for folder in args.dirs:
        for fp in sorted(glob.glob(os.path.join(folder, "*.json"))):
            issues = audit_file(fp, engine)
            all_issues.extend(issues)

    if not all_issues:
        print("No policy violations found.")
        return

    # Group for display
    relabels = [i for i in all_issues if i["action"] != "remove"]
    removes = [i for i in all_issues if i["action"] == "remove"]

    print(
        f"\nFound {len(relabels)} relabels and {len(removes)} removals across "
        f"{len(set(i['file'] for i in all_issues))} file(s).\n"
    )

    if relabels:
        # Summarise by (text, current→expected)
        counts: dict[tuple, int] = defaultdict(int)
        for i in relabels:
            counts[(i["text"], i["current"], i["action"])] += 1
        print("Relabels (text | current → new | count):")
        for (text, current, action), n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {text:25s}  {current:8s} → {action.split('→')[1]:8s}  ×{n}")

    if removes:
        counts2: dict[tuple, int] = defaultdict(int)
        for i in removes:
            counts2[(i["text"], i["current"])] += 1
        print("\nRemovals (text | current label | count):")
        for (text, current), n in sorted(counts2.items(), key=lambda x: -x[1]):
            print(f"  {text:25s}  ({current})  ×{n}")

    if args.apply:
        print("\nApplying fixes…")
        apply_fixes(all_issues)
        print("\nDone. Regenerate data/augmented/ before retraining.")
    else:
        print("\n[Dry run] Pass --apply to write changes.")


if __name__ == "__main__":
    main()

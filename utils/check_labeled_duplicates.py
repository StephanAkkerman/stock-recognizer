"""Check for duplicate labeled tasks by text in Label Studio exports.

Usage examples:
  python utils/check_labeled_duplicates.py            # scans data/labeled/
  python utils/check_labeled_duplicates.py --files data/labeled/new_batch.json
  python utils/check_labeled_duplicates.py --dir data/labeled --report dup.csv

The script normalizes whitespace when comparing texts. If you pass --files,
the script will highlight duplicates between those files and the rest of the
directory (useful for checking a new batch against existing labels).
"""
from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def load_tasks(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected list in {path}")
    return data


def norm_text(text: str) -> str:
    return " ".join(text.split()).strip()


def scan_dir(dirpath: Path) -> Dict[str, List[Tuple[Path, int, int]]]:
    """Return mapping normalized_text -> list of (file, index, task_id).
    """
    mapping: Dict[str, List[Tuple[Path, int, int]]] = {}
    for p in sorted(dirpath.glob("*.json")):
        try:
            tasks = load_tasks(p)
        except Exception as e:
            print(f"skipping {p}: {e}")
            continue
        for i, t in enumerate(tasks):
            text = t.get("data", {}).get("text", "")
            n = norm_text(text)
            entry = (p, i, t.get("id"))
            mapping.setdefault(n, []).append(entry)
    return mapping


def report_duplicates(mapping: Dict[str, List[Tuple[Path, int, int]]], new_files: List[Path] | None = None) -> List[Tuple[str, List[Tuple[Path, int, int]]]]:
    """Return list of (text, occurrences) for duplicates.

    If new_files is provided, mark duplicates that involve at least one path in new_files.
    """
    results = []
    new_set = set(new_files) if new_files else None
    for text, occ in mapping.items():
        if len(occ) > 1:
            if new_set is None:
                results.append((text, occ))
            else:
                # only include if at least one occurrence is in new_files
                if any(o[0] in new_set for o in occ):
                    results.append((text, occ))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/labeled", help="directory with labeled json files")
    ap.add_argument("--files", nargs="*", help="specific files to treat as 'new' (relative to cwd)")
    ap.add_argument("--report", help="CSV path to write duplicate report")
    ap.add_argument("--show-sample", type=int, default=5, help="how many duplicate examples to show")
    args = ap.parse_args()

    base = Path(args.dir)
    if not base.exists():
        print(f"directory not found: {base}")
        raise SystemExit(1)

    new_files = None
    if args.files:
        new_files = [Path(f) for f in args.files]

    mapping = scan_dir(base)
    duplicates = report_duplicates(mapping, new_files)

    total_dupes = sum(len(occ) for _, occ in duplicates)
    print(f"found {len(duplicates)} distinct duplicate texts, {total_dupes} total occurrences")

    if args.report:
        with open(args.report, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["text", "file", "index", "task_id"])
            for text, occ in duplicates:
                for p, i, tid in occ:
                    w.writerow([text, str(p), i, tid])
        print(f"wrote CSV report to {args.report}")

    # Print a concise sample
    shown = 0
    for text, occ in duplicates:
        if shown >= args.show_sample:
            break
        print("---")
        print(text)
        for p, i, tid in occ:
            print(f"  - {p} [index={i} task_id={tid}]")
        shown += 1

    if not duplicates:
        print("no duplicates found")


if __name__ == "__main__":
    main()

"""Fix two systematic annotation issues across Label Studio exports.

1. Possessives: entity spans ending in "'s" or "'" are trimmed to the bare name.
   e.g. "Tesla's" → "Tesla", "$GRAB's" → "$GRAB", "Robinhood'" → "Robinhood"

2. Label normalization: entity texts in NORMALIZE_LABEL are pinned to a
   canonical label regardless of how they were annotated.
   e.g. GME and AMC always "ticker", never "company".

Usage:
    python utils/fix_annotations.py                         # dry-run: preview only
    python utils/fix_annotations.py --apply                 # write files in-place
    python utils/fix_annotations.py --folders data/labeled  # one folder only
"""

import argparse
import glob
import json
import os
import shutil
from collections import defaultdict

# Entities whose label should be normalized to a single canonical value.
# Add entries here whenever error_analysis reveals systematic ticker↔company
# confusion for a specific token.
NORMALIZE_LABEL = {
    "GME": "ticker",
    "AMC": "ticker",
}

# Use chr() so these survive editors that silently convert quote characters.
_RQUOTE = chr(0x2019)  # RIGHT SINGLE QUOTATION MARK  (most common in pasted text)
_LQUOTE = chr(0x2018)  # LEFT SINGLE QUOTATION MARK
_APOS = chr(39)        # APOSTROPHE U+0027 (plain ASCII)


def _norm_apos(text):
    """Replace all apostrophe variants with plain ASCII for uniform comparisons."""
    return text.replace(_RQUOTE, _APOS).replace(_LQUOTE, _APOS)


# Company names whose apostrophe is part of the brand, not a possessive suffix.
# Built with _APOS so the set uses consistent ASCII regardless of editor encoding.
# Add here whenever the stripper would corrupt a legitimate brand name.
_a = _APOS
POSSESSIVE_EXEMPT = {
    "wendy" + _a + "s",
    "mcdonald" + _a + "s",
    "papa john" + _a + "s",
    "papa john" + _a + "s international",
    "denny" + _a + "s",
    "macy" + _a + "s",
    "levi" + _a + "s",
    "kellogg" + _a + "s",
    "caesar" + _a + "s",
    "caesars",
    "hardee" + _a + "s",
    "carl" + _a + "s jr",
    "arby" + _a + "s",
}


def _strip_possessive(text):
    """Return (stripped, chars_removed). Leaves text unchanged if no suffix matches,
    if stripping would leave an empty string, or if the text is an exempt brand name.

    All comparisons run on apostrophe-normalised, lowercased text so ASCII and
    curly apostrophes are treated identically. Trimming uses char count, which
    is the same for all apostrophe variants (each is a single Unicode code point).
    """
    norm = _norm_apos(text).lower()
    if norm in POSSESSIVE_EXEMPT:
        return text, 0
    for suffix in (_APOS + "s", _APOS):
        if norm.endswith(suffix) and len(norm) > len(suffix):
            n = len(suffix)
            return text[:-n], n
    return text, 0


def _fix_task(task, full_text):
    """Apply fixes to one task's annotations in-place.

    Returns a list of human-readable change descriptions (empty if nothing changed).
    """
    changes = []
    if not task.get("annotations"):
        return changes

    annotation = task["annotations"][0]
    if annotation.get("was_cancelled"):
        return changes

    for r in annotation.get("result", []):
        if r.get("type") != "labels":
            continue

        val = r["value"]
        start = val["start"]
        end = val["end"]
        label = val["labels"][0]
        entity_text = full_text[start:end]

        # 1. Possessive stripping
        stripped, n_removed = _strip_possessive(entity_text)
        if n_removed:
            changes.append(
                f"    possessive  [{entity_text!r}] ({label}) → [{stripped!r}]"
            )
            val["end"] = end - n_removed
            if "text" in val:
                val["text"] = stripped
            entity_text = stripped

        # 2. Label normalization (match against bare symbol, case-insensitive)
        bare = entity_text.lstrip("$").upper()
        canonical = NORMALIZE_LABEL.get(bare)
        if canonical and label != canonical:
            changes.append(
                f"    label norm  [{entity_text!r}] {label!r} → {canonical!r}"
            )
            val["labels"] = [canonical]

    return changes


def _process_file(fp, apply):
    """Load, fix, and optionally save one JSON file.

    Returns the number of individual changes made.
    """
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    file_changes = []
    for task in data:
        full_text = task.get("data", {}).get("text", "")
        task_changes = _fix_task(task, full_text)
        if task_changes:
            file_changes.append((task.get("id", "?"), task_changes))

    if not file_changes:
        return 0

    tag = "[APPLY]" if apply else "[DRY RUN]"
    print(f"\n{tag} {fp}")
    for task_id, task_changes in file_changes:
        print(f"  task {task_id}:")
        for line in task_changes:
            print(line)

    if apply:
        shutil.copy2(fp, fp + ".bak")
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return sum(len(c) for _, c in file_changes)


def _process_folder(folder, apply):
    files = sorted(glob.glob(os.path.join(folder, "*.json")))
    total = 0
    files_touched = 0
    for fp in files:
        if "augmented_" in os.path.basename(fp):
            continue  # skip augmented — they'll be regenerated from fixed sources
        n = _process_file(fp, apply)
        if n:
            total += n
            files_touched += 1
    return total, files_touched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folders",
        nargs="+",
        default=["data/labeled", "data/test"],
        help="Folders to scan (default: data/labeled data/test)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes to disk; default is dry-run only",
    )
    args = parser.parse_args()

    total_changes = 0
    total_files = 0
    for folder in args.folders:
        if not os.path.isdir(folder):
            print(f"[skip] {folder} not found")
            continue
        n_changes, n_files = _process_folder(folder, args.apply)
        total_changes += n_changes
        total_files += n_files

    verb = "Applied" if args.apply else "Would apply"
    print(f"\n{verb} {total_changes} fix(es) across {total_files} file(s).")
    if not args.apply and total_changes > 0:
        print("Re-run with --apply to write changes.")


if __name__ == "__main__":
    main()

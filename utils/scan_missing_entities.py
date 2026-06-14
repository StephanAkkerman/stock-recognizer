"""Scan labeled/test/preds JSON files for entity mentions that match a
known entity by one of four narrow rules:

  1. Punctuation-stripped exact match — apostrophe/comma variants:
       "Wendy's" ↔ "Wendys",  "Meta's" ↔ "Metas"
  2. Trailing-'s' match — possessive/plural:
       "Foxconn" ↔ "Foxconns",  "Nvidia" ↔ "Nvidias"
  3. Token one char longer, edit-distance 1 — typo in labeled entity:
       "Antropic" ↔ "Anthropic",  "Discovey" ↔ "Discovery"
  4. Strip trailing 's' then edit-distance ≤ 1 — the Nvdias/Nvidia pattern:
       "Nvdias" → strip s → "Nvdia" → edit-dist 1 from "Nvidia"

These rules are narrow enough to produce zero false positives in practice,
so the output can be auto-fixed with --fix without manual review.

Usage:
    python utils/scan_missing_entities.py          # dry-run report
    python utils/scan_missing_entities.py --fix    # patch preds files
    python utils/scan_missing_entities.py --fix --fix-reviewed  # patch all
"""

import argparse
import collections
import json
import os
import re
import uuid

from rich.console import Console
from rich.table import Table

console = Console()

ENTITY_SOURCES = [
    "data/labeled/labeled_100.json",
    "data/labeled/labeled_200.json",
    "data/labeled/labeled_300.json",
    "data/labeled/labeled_auto_labeled.json",
    "data/labeled/labeled_final.json",
    "data/test/test_labeled_100.json",
    "data/test/test_labeled_200.json",
    "data/test/test_labeled_300.json",
    "data/test/test_labeled_auto_labeled.json",
    "data/test/test_labeled_final.json",
    "data/preds/auto_labeled.json",
    "data/preds/batch_final_prelabeled.json",
    "data/preds/train_base_100.json",
    "data/preds/train_base_200.json",
    "data/preds/train_base_300.json",
]

SCAN_TARGETS = {
    "preds": [
        "data/preds/auto_labeled.json",
        "data/preds/batch_final_prelabeled.json",
        "data/preds/train_base_100.json",
        "data/preds/train_base_200.json",
        "data/preds/train_base_300.json",
    ],
    "reviewed": [
        "data/labeled/labeled_100.json",
        "data/labeled/labeled_200.json",
        "data/labeled/labeled_300.json",
        "data/labeled/labeled_auto_labeled.json",
        "data/labeled/labeled_final.json",
        "data/test/test_labeled_100.json",
        "data/test/test_labeled_200.json",
        "data/test/test_labeled_300.json",
        "data/test/test_labeled_auto_labeled.json",
        "data/test/test_labeled_final.json",
    ],
}

_NONALPHA = re.compile(r"[^a-z0-9]")


# ---------------------------------------------------------------------------
# Rule helpers
# ---------------------------------------------------------------------------


def _levenshtein(s, t):
    """Standard DP edit distance (insert/delete/substitute = 1 each)."""
    if len(s) < len(t):
        return _levenshtein(t, s)
    if not t:
        return len(s)
    prev = list(range(len(t) + 1))
    for c in s:
        curr = [prev[0] + 1]
        for j, d in enumerate(t):
            curr.append(min(prev[j] + (c != d), curr[-1] + 1, prev[j + 1] + 1))
        prev = curr
    return prev[-1]


def _matches(ent_lower, tok_lower):
    """True when tok_lower is a near-miss variant of ent_lower.

    Both arguments must already be lowercased.  Returns False for identical
    strings (exact match is handled upstream) and for strings that differ in
    more fundamental ways than the four targeted rules allow.
    """
    if ent_lower == tok_lower:
        return False

    # Rule 1: punctuation/apostrophe stripped exact match
    e_a = _NONALPHA.sub("", ent_lower)
    t_a = _NONALPHA.sub("", tok_lower)
    if e_a and t_a and e_a == t_a:
        return True

    # Rule 2: one string is the other + trailing 's'
    if tok_lower == ent_lower + "s" or ent_lower == tok_lower + "s":
        return True

    # Rule 3: token is one char longer (entity has a missing/wrong char)
    if len(tok_lower) == len(ent_lower) + 1 and _levenshtein(ent_lower, tok_lower) == 1:
        return True

    # Rule 4: strip trailing 's' from token then edit-dist ≤ 1 (Nvdias/Nvidia)
    if tok_lower.endswith("s") and len(tok_lower) >= 5:
        stripped = tok_lower[:-1]
        # Avoid re-triggering Rule 2 (already covered)
        if stripped != ent_lower and _levenshtein(stripped, ent_lower) <= 1:
            return True

    return False


# ---------------------------------------------------------------------------
# Entity dictionary
# ---------------------------------------------------------------------------


def _load_tasks(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _iter_results(task):
    """Yield result dicts from whichever annotation block exists in `task`."""
    for ann in task.get("annotations") or task.get("predictions") or []:
        yield from ann.get("result", [])


def build_entity_dict(source_paths):
    """Return {ent_text: set_of_labels} from all annotation results."""
    entities = {}
    for path in source_paths:
        for task in _load_tasks(path):
            for r in _iter_results(task):
                v = r.get("value", {})
                text = (v.get("text") or "").strip()
                labels = v.get("labels", [])
                if text and labels:
                    entities.setdefault(text, set()).update(labels)
    return entities


def build_lookup(entity_dict, min_len=6):
    """Build fast lookup structures for the four rules.

    Returns:
        by_alpha  — {alpha_str: [(ent_text, labels)]}   for Rule 1
        by_lower  — {ent_lower: (ent_text, labels)}     for Rule 2
        by_len    — {int: [(ent_text, labels)]}          for Rules 3 & 4
    """
    by_alpha = collections.defaultdict(list)
    by_lower = {}
    by_len = collections.defaultdict(list)
    for text, labels in entity_dict.items():
        if len(text) < min_len:
            continue
        low = text.lower()
        by_lower[low] = (text, labels)
        by_alpha[_NONALPHA.sub("", low)].append((text, labels))
        by_len[len(text)].append((text, labels))
    return by_alpha, by_lower, by_len


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


def _labeled_intervals(task):
    intervals = []
    for r in _iter_results(task):
        v = r.get("value", {})
        s, e = v.get("start"), v.get("end")
        if s is not None and e is not None:
            intervals.append((s, e))
    return intervals


def _append_result(task, start, end, text, label):
    """Append a new result span to the task's annotation/prediction block."""
    ann_list = task.get("annotations") or task.get("predictions")
    if not ann_list:
        return
    if "annotations" in task:
        ann_list[0]["result"].append(_make_result(start, end, text, label))
    else:
        # predictions format omits id and value.text
        ann_list[0]["result"].append(
            {
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "value": {"start": start, "end": end, "labels": [label]},
            }
        )


def _overlaps(start, end, intervals):
    return any(start < ie and is_ < end for is_, ie in intervals)


def _region_id():
    return uuid.uuid4().hex[:10]


def _make_result(start, end, text, label):
    return {
        "id": _region_id(),
        "from_name": "label",
        "to_name": "text",
        "type": "labels",
        "value": {"start": start, "end": end, "text": text, "labels": [label]},
    }


def _find_hits(post_text, post_lower, existing, by_alpha, by_lower, by_len):
    """Yield (start, end, tok_text, ent_text, labels) for every match."""
    tokens = list(re.finditer(r"\S+", post_text))
    seen_positions = set()

    for tok_m in tokens:
        tok_text = tok_m.group()
        # Only proper nouns: must start with uppercase letter.
        if not tok_text[0].isupper():
            continue
        tok_lower = tok_text.lower()
        tok_len = len(tok_text)
        if tok_len < 5:
            continue
        start, end = tok_m.start(), tok_m.end()
        if _overlaps(start, end, existing) or (start, end) in seen_positions:
            continue

        matched_ent = None

        # Rule 1 — alpha match (O(1) lookup)
        tok_alpha = _NONALPHA.sub("", tok_lower)
        for ent_text, labels in by_alpha.get(tok_alpha, []):
            e_lower = ent_text.lower()
            if e_lower == tok_lower:
                continue  # exact match — not a normalisation mismatch
            if e_lower in post_lower:
                continue  # already findable by exact path
            matched_ent = (ent_text, labels)
            break

        if not matched_ent:
            # Rule 2a — token = entity + 's'  →  look up tok[:-1]
            if tok_lower.endswith("s"):
                stripped = tok_lower[:-1]
                entry = by_lower.get(stripped)
                if entry:
                    ent_text, labels = entry
                    if ent_text.lower() not in post_lower:
                        matched_ent = (ent_text, labels)

        if not matched_ent:
            # Rule 2b — entity = token + 's'  →  look up tok + 's'
            entry = by_lower.get(tok_lower + "s")
            if entry:
                ent_text, labels = entry
                if ent_text.lower() not in post_lower:
                    matched_ent = (ent_text, labels)

        if not matched_ent:
            # Rule 3 — entity is tok_len-1 chars, edit dist 1.
            # Require entity length ≥ 7 to avoid matching short pairs like
            # "AMD's"/"AMTD's" or "Webull"/"Weibull" that are different entities.
            for ent_text, labels in by_len.get(tok_len - 1, []):
                if len(ent_text) < 7:
                    continue
                e_lower = ent_text.lower()
                if e_lower in post_lower:
                    continue
                if _levenshtein(e_lower, tok_lower) == 1:
                    matched_ent = (ent_text, labels)
                    break

        if not matched_ent and tok_lower.endswith("s") and tok_len >= 5:
            # Rule 4 — strip 's', edit dist ≤ 1 against entities of similar length
            stripped = tok_lower[:-1]
            for ent_len in (len(stripped) - 1, len(stripped), len(stripped) + 1):
                for ent_text, labels in by_len.get(ent_len, []):
                    e_lower = ent_text.lower()
                    if e_lower in post_lower:
                        continue
                    if e_lower == stripped:
                        continue  # Rule 2a already handles this
                    if _levenshtein(stripped, e_lower) <= 1:
                        matched_ent = (ent_text, labels)
                        break
                if matched_ent:
                    break

        if matched_ent:
            seen_positions.add((start, end))
            ent_text, labels = matched_ent
            yield start, end, tok_text, ent_text, labels


def scan_file(path, by_alpha, by_lower, by_len, fix=False):
    """Scan one JSON file; return list of hit dicts.  Patches in-place when fix=True."""
    tasks = _load_tasks(path)
    hits = []
    modified = False

    for task in tasks:
        post_text = task.get("data", {}).get("text", "")
        post_lower = post_text.lower()
        existing = _labeled_intervals(task)

        for start, end, tok_text, ent_text, labels in _find_hits(
            post_text, post_lower, existing, by_alpha, by_lower, by_len
        ):
            for label in labels:
                hits.append(
                    {
                        "task_id": task.get("id"),
                        "entity_name": ent_text,
                        "matched_text": tok_text,
                        "start": start,
                        "end": end,
                        "label": label,
                        "text_excerpt": post_text[max(0, start - 40) : end + 40],
                    }
                )
            if fix:
                for label in labels:
                    _append_result(task, start, end, tok_text, label)
                existing.append((start, end))
                modified = True

    if fix and modified:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)

    return hits


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(all_hits_by_file):
    total = sum(len(h) for h in all_hits_by_file.values())
    if total == 0:
        console.print("[green]No missing entities found.[/green]")
        return

    console.print(
        f"\n[bold yellow]Found {total} potential missed entity span(s):[/bold yellow]\n"
    )
    for path, hits in all_hits_by_file.items():
        if not hits:
            continue
        short = os.path.basename(path)
        table = Table(title=f"{short} ({len(hits)} hit(s))", show_lines=True)
        table.add_column("Task ID", style="cyan", no_wrap=True)
        table.add_column("Known as", style="green")
        table.add_column("Found in text", style="yellow")
        table.add_column("Label")
        table.add_column("Rule")
        table.add_column("Context (~80 chars)")
        for h in hits:
            excerpt = h["text_excerpt"].replace("\n", " ")
            rel_start = h["start"] - max(0, h["start"] - 40)
            rel_end = rel_start + (h["end"] - h["start"])
            highlighted = (
                excerpt[:rel_start]
                + f"[bold red]{excerpt[rel_start:rel_end]}[/bold red]"
                + excerpt[rel_end:]
            )
            rule = _which_rule(h["entity_name"].lower(), h["matched_text"].lower())
            table.add_row(
                str(h["task_id"]),
                h["entity_name"],
                h["matched_text"],
                h["label"],
                rule,
                highlighted,
            )
        console.print(table)


def _which_rule(e, t):
    if _NONALPHA.sub("", e) == _NONALPHA.sub("", t) and e != t:
        return "punct"
    if t == e + "s" or e == t + "s":
        return "trailing-s"
    if len(t) == len(e) + 1 and _levenshtein(e, t) == 1:
        return "edit-1"
    return "edit-1+s"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Patch preds JSON files in-place with the discovered spans.",
    )
    parser.add_argument(
        "--fix-reviewed",
        action="store_true",
        help="Also patch labeled/ and test/ files (use with caution).",
    )
    args = parser.parse_args()

    console.print("[bold cyan]Building entity dictionary...[/bold cyan]")
    entity_dict = build_entity_dict(ENTITY_SOURCES)
    console.print(f"  {len(entity_dict)} unique entity texts loaded.")

    by_alpha, by_lower, by_len = build_lookup(entity_dict)

    targets = list(SCAN_TARGETS["preds"])
    if args.fix_reviewed or not args.fix:
        targets += SCAN_TARGETS["reviewed"]

    all_hits = {}
    total_tasks = 0
    for path in targets:
        tasks = _load_tasks(path)
        if not tasks:
            continue
        total_tasks += len(tasks)
        do_fix = args.fix and (path in SCAN_TARGETS["preds"] or args.fix_reviewed)
        console.print(f"  Scanning [dim]{path}[/dim] ({len(tasks)} tasks)...")
        hits = scan_file(path, by_alpha, by_lower, by_len, fix=do_fix)
        if hits:
            all_hits[path] = hits

    console.print(f"\nScanned {total_tasks} tasks across {len(targets)} file(s).")
    print_report(all_hits)

    if args.fix:
        fixed = [p for p in all_hits if p in SCAN_TARGETS["preds"] or args.fix_reviewed]
        if fixed:
            console.print(f"\n[green]Patched {len(fixed)} file(s) in-place.[/green]")


if __name__ == "__main__":
    main()

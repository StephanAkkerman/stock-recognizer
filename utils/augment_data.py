import copy
import glob
import json
import os
import random
import re

import financedatabase as fd

from stock_recognizer.constants import (
    AMBIGUOUS_WORDS,
    EXCHANGE_BLACKLIST,
    US_MAJOR_EXCHANGES,
)


LABEL_TYPES = ("ticker", "company")

# How often to draw replacements from the broad financedatabase pool versus
# the small pool harvested from human labels. 0.7 keeps labeled entities
# heavily over-represented per-item (since the labeled pool is ~10-100x
# smaller) while still exposing the model to unseen real tickers.
# Lower this if recall on the entities you care about drops; raise it if
# the model is over-fitting to the labeled vocabulary.
EXPANDED_POOL_WEIGHT = 0.7

# Legal/structural suffixes stripped before extracting a "first significant
# word" company short-form. Reddit users write "Microsoft", not "Microsoft
# Corporation" — matching that distribution matters more than completeness.
_COMPANY_SUFFIX_TOKENS = {
    "INC", "CORP", "CORPORATION", "LTD", "LIMITED", "PLC", "CO",
    "COMPANY", "HOLDINGS", "GROUP", "AG", "SA", "NV", "LLC", "LP",
    "THE", "&",
}


def build_replacement_pool_from_labels(folder_path):
    """
    Scans all original human-labeled JSON files to extract a dynamic pool
    of unique tickers and companies based on actual annotations.
    """
    files = glob.glob(os.path.join(folder_path, "*.json"))
    original_files = [f for f in files if "augmented_" not in os.path.basename(f)]

    pool = {label: set() for label in LABEL_TYPES}

    for file_path in original_files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                ls_data = json.load(f)
            except json.JSONDecodeError:
                continue

        for task in ls_data:
            if not task.get("annotations") or task["annotations"][0].get(
                "was_cancelled"
            ):
                continue

            text = task["data"]["text"]
            results = task["annotations"][0].get("result", [])

            for r in results:
                if r.get("type") != "labels":
                    continue
                val = r["value"]
                label = val["labels"][0]
                entity_text = text[val["start"] : val["end"]].strip()

                if label not in pool or not entity_text:
                    continue

                # Cashtags are re-applied at swap time, so the pool stores bare symbols.
                if label == "ticker" and entity_text.startswith("$"):
                    entity_text = entity_text[1:]
                pool[label].add(entity_text)

    return {k: sorted(v) for k, v in pool.items()}


def _clean_company_short_form(name):
    """Extract a Reddit-style short company name (e.g., 'Microsoft Corporation' → 'Microsoft').

    Returns None if no usable short form can be recovered. Length-2 tokens
    are rejected because they typically clash with tickers (AT, BP, GM).
    """
    if not isinstance(name, str):
        return None
    cleaned = re.sub(r"[.,&]", " ", name)
    for tok in cleaned.split():
        if tok.upper() in _COMPANY_SUFFIX_TOKENS:
            continue
        if len(tok) <= 2:
            continue
        # Reject tokens that are mostly digits/punctuation (e.g. "000%", "1847",
        # "1RT"). Real Reddit-style company mentions need a substantive word.
        if sum(1 for c in tok if c.isalpha()) < 3:
            continue
        return tok
    return None


def build_financedatabase_pool():
    """Pull a US-only pool of tickers and short company names from financedatabase.

    Mirrors the inference-time filter in ``stock_recognizer.engine``:
    US major exchanges only, exchange-suffix blacklist applied, and entries
    that collide with ``AMBIGUOUS_WORDS`` dropped so the model isn't trained
    to recognise tokens the regex path will reject anyway.
    """
    market = fd.Equities().select(exchange=list(US_MAJOR_EXCHANGES))

    tickers = set()
    companies = set()
    for ticker, row in market.iterrows():
        if not isinstance(ticker, str):
            continue
        if any(ext in ticker for ext in EXCHANGE_BLACKLIST):
            continue
        if ticker.upper() in AMBIGUOUS_WORDS:
            continue
        tickers.add(ticker)

        short = _clean_company_short_form(row.get("name"))
        if short and short.upper() not in AMBIGUOUS_WORDS:
            companies.add(short)

    return {"ticker": sorted(tickers), "company": sorted(companies)}


def _sample_replacement(label, labeled_pool, expanded_pool, original_bare):
    """Pick a non-trivial replacement, biased toward `expanded_pool` per `EXPANDED_POOL_WEIGHT`.

    Falls back to whichever pool has data if the preferred source is empty
    for this label, so a missing expanded pool never blocks augmentation.
    """
    prefer_expanded = random.random() < EXPANDED_POOL_WEIGHT
    primary = expanded_pool.get(label, []) if prefer_expanded else labeled_pool.get(label, [])
    secondary = labeled_pool.get(label, []) if prefer_expanded else expanded_pool.get(label, [])

    source = primary or secondary
    if not source:
        return None
    alternatives = [c for c in source if c != original_bare]
    if not alternatives:
        return None
    return random.choice(alternatives)


def _resolve_overlaps(label_results):
    """Resolves overlapping annotations into independent swap groups.

    Returns a list of ``(start, end, [annotations])`` tuples — one per unique
    span that will receive a single swap. Annotations sharing a span (e.g.
    "BP" labeled as both ticker and company) are grouped so they receive the
    same replacement. Annotations contained within a larger span are dropped
    from the output entirely (we lose that single label but the outer span
    stays consistent with the swapped text).

    Also returns the set of dropped annotation ids so the caller can remove
    them from the augmented task. The second return value is ``None`` if the
    task has unresolvable overlaps (partial non-containment) and should be
    skipped.
    """
    by_span = {}
    for r in label_results:
        v = r["value"]
        key = (v["start"], v["end"])
        by_span.setdefault(key, []).append(r)

    spans = sorted(by_span.keys())
    keep = set(spans)

    # Drop spans strictly contained within another. ``s1 <= s2 and e2 <= e1``
    # with the spans non-equal means span2 is inside span1.
    for s1, e1 in spans:
        for s2, e2 in spans:
            if (s1, e1) == (s2, e2):
                continue
            if s1 <= s2 and e2 <= e1:
                keep.discard((s2, e2))

    kept_sorted = sorted(keep)

    # After dropping containments, any remaining overlap is a "weird" partial
    # overlap (e.g. (10,15) and (12,18)). Bail on the whole task — these are
    # rare and not worth the complexity to repair.
    last_end = -1
    for s, e in kept_sorted:
        if s < last_end:
            return None, None
        last_end = e

    groups = [(s, e, by_span[(s, e)]) for s, e in kept_sorted]
    kept_ids = {id(r) for _, _, rs in groups for r in rs}
    dropped_ids = {id(r) for r in label_results if id(r) not in kept_ids}
    return groups, dropped_ids


def augment_task(task, labeled_pool, expanded_pool=None):
    """Swaps labeled entities from back-to-front to preserve index integrity.

    Replacement candidates are drawn from `labeled_pool` and `expanded_pool`
    with weight `EXPANDED_POOL_WEIGHT` favouring the broad financedatabase
    pool. Pass `expanded_pool=None` to fall back to the legacy labeled-only
    behaviour (e.g. for tests).
    """
    if expanded_pool is None:
        expanded_pool = {}

    augmented_task = copy.deepcopy(task)
    annotation = augmented_task["annotations"][0]
    results = annotation.get("result", [])

    label_results = [r for r in results if r.get("type") == "labels"]

    groups, dropped_ids = _resolve_overlaps(label_results)
    if groups is None:
        return None

    # Process from rightmost span to leftmost so earlier indices stay valid
    # as we mutate the text in place.
    groups.sort(key=lambda g: g[0], reverse=True)

    text = augmented_task["data"]["text"]
    changed = False

    for start, end, group in groups:
        # All annotations in this group share the span; the label they use to
        # pick a replacement comes from the first one. (For multi-label spans
        # like "BP" being both ticker and company, we just pick one pool.)
        primary = group[0]
        label = primary["value"]["labels"][0]

        original_word = text[start:end]
        original_bare = original_word.lstrip("$")

        replacement = _sample_replacement(
            label, labeled_pool, expanded_pool, original_bare
        )
        if replacement is None:
            continue

        if original_word.startswith("$") and not replacement.startswith("$"):
            replacement = "$" + replacement

        text = text[:start] + replacement + text[end:]
        len_diff = len(replacement) - len(original_word)
        new_end = start + len(replacement)

        for r in group:
            r["value"]["end"] = new_end
            if "text" in r["value"]:
                r["value"]["text"] = replacement
        changed = True

        # Shift every annotation strictly to the right of this span. Group
        # members share start == ``start`` so they aren't shifted; spans
        # already processed sit to the right and need their offsets updated.
        for other_r in label_results:
            other_val = other_r["value"]
            if other_val["start"] > start:
                other_val["start"] += len_diff
                other_val["end"] += len_diff

    if not changed:
        return None

    # Drop contained annotations; the surviving outer span already covers
    # the region and its text has been swapped.
    if dropped_ids:
        annotation["result"] = [
            r for r in results
            if r.get("type") != "labels" or id(r) not in dropped_ids
        ]

    augmented_task["data"]["text"] = text

    # Predictions reference the pre-swap text; their offsets are now stale.
    # Drop them so downstream consumers can't accidentally train on bad spans.
    annotation["prediction"] = {}
    annotation["parent_prediction"] = None
    augmented_task["predictions"] = []

    return augmented_task


def run_augmentation(
    source_folder, output_folder, multiplier=3, seed=None, use_expanded_pool=True
):
    """Builds replacement pools from `source_folder` (plus financedatabase if
    `use_expanded_pool=True`) and writes synthetic variations into `output_folder`.

    Default multiplier dropped to 3 — with a ~10x larger entity pool, every
    pass already covers much more vocabulary than the old 5× labeled-only
    setup, so additional passes mostly inflate training time.
    """
    if seed is not None:
        random.seed(seed)

    os.makedirs(output_folder, exist_ok=True)

    print("Scanning labeled dataset to build replacement pool...")
    labeled_pool = build_replacement_pool_from_labels(source_folder)
    print(f"  Labeled tickers  : {len(labeled_pool['ticker'])}")
    print(f"  Labeled companies: {len(labeled_pool['company'])}")

    expanded_pool = {}
    if use_expanded_pool:
        print("Loading financedatabase pool (US majors, blacklist-filtered)...")
        expanded_pool = build_financedatabase_pool()
        print(f"  Expanded tickers  : {len(expanded_pool['ticker'])}")
        print(f"  Expanded companies: {len(expanded_pool['company'])}")
        print(f"  Mix weight: {EXPANDED_POOL_WEIGHT:.0%} expanded / "
              f"{1 - EXPANDED_POOL_WEIGHT:.0%} labeled")

    if (len(labeled_pool["ticker"]) + len(expanded_pool.get("ticker", []))) < 2 or (
        len(labeled_pool["company"]) + len(expanded_pool.get("company", []))
    ) < 2:
        print("Error: Not enough unique labels found to safely perform swaps.")
        return

    files = glob.glob(os.path.join(source_folder, "*.json"))
    original_files = [f for f in files if "augmented_" not in os.path.basename(f)]

    total_generated = 0
    total_skipped = 0

    for file_path in original_files:
        with open(file_path, "r", encoding="utf-8") as f:
            ls_data = json.load(f)

        file_filename = os.path.basename(file_path)

        for i in range(multiplier):
            augmented_batch = []
            for task in ls_data:
                if not task.get("annotations") or task["annotations"][0].get(
                    "was_cancelled"
                ):
                    continue
                try:
                    aug_task = augment_task(task, labeled_pool, expanded_pool)
                except Exception as exc:
                    print(f"  ! skipped task {task.get('id')} in {file_filename}: {exc}")
                    total_skipped += 1
                    continue

                if aug_task is None:
                    total_skipped += 1
                    continue

                augmented_batch.append(aug_task)
                total_generated += 1

            output_name = os.path.join(output_folder, f"augmented_m{i}_{file_filename}")
            with open(output_name, "w", encoding="utf-8") as out_f:
                json.dump(augmented_batch, out_f, indent=2, ensure_ascii=False)

    print(
        f"Generated {total_generated} augmented samples across {multiplier} variants "
        f"({total_skipped} skipped). Wrote to {output_folder}."
    )


if __name__ == "__main__":
    labeled_folder = "data/labeled"
    augmented_folder = "data/augmented"
    run_augmentation(labeled_folder, augmented_folder, multiplier=3)

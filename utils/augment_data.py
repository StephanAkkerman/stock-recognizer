import copy
import glob
import json
import os
import random


LABEL_TYPES = ("ticker", "company")


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


def augment_task(task, pool):
    """Swaps labeled entities from back-to-front to preserve index integrity."""
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

        candidates = pool.get(label)
        if not candidates:
            continue

        original_word = text[start:end]
        original_bare = original_word.lstrip("$")

        # Avoid no-op swaps that just re-emit the original token.
        alternatives = [c for c in candidates if c != original_bare]
        if not alternatives:
            continue
        replacement = random.choice(alternatives)

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


def run_augmentation(source_folder, output_folder, multiplier=5, seed=None):
    """Builds a dynamic pool from labels in `source_folder` and writes synthetic
    variations into `output_folder`."""
    if seed is not None:
        random.seed(seed)

    os.makedirs(output_folder, exist_ok=True)

    print("Scanning dataset to build dynamic replacement pool...")
    dynamic_pool = build_replacement_pool_from_labels(source_folder)

    print(f"Discovered unique tickers  : {len(dynamic_pool['ticker'])}")
    print(f"Discovered unique companies: {len(dynamic_pool['company'])}")

    if len(dynamic_pool["ticker"]) < 2 or len(dynamic_pool["company"]) < 2:
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
                    aug_task = augment_task(task, dynamic_pool)
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
    run_augmentation(labeled_folder, augmented_folder, multiplier=5)

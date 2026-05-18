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


def augment_task(task, pool):
    """Swaps labeled entities from back-to-front to preserve index integrity."""
    augmented_task = copy.deepcopy(task)
    annotation = augmented_task["annotations"][0]
    results = annotation.get("result", [])

    label_results = [r for r in results if r.get("type") == "labels"]
    label_results.sort(key=lambda x: x["value"]["start"], reverse=True)

    text = augmented_task["data"]["text"]
    changed = False

    for r in label_results:
        val = r["value"]
        label = val["labels"][0]

        candidates = pool.get(label)
        if not candidates:
            continue

        start = val["start"]
        end = val["end"]
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

        val["end"] = start + len(replacement)
        if "text" in val:
            val["text"] = replacement
        changed = True

        for other_r in label_results:
            other_val = other_r["value"]
            if other_val["start"] > start:
                other_val["start"] += len_diff
                other_val["end"] += len_diff

    if not changed:
        return None

    augmented_task["data"]["text"] = text

    # Predictions reference the pre-swap text; their offsets are now stale.
    # Drop them so downstream consumers can't accidentally train on bad spans.
    annotation["prediction"] = {}
    annotation["parent_prediction"] = None
    augmented_task["predictions"] = []

    return augmented_task


def run_augmentation(folder_path, multiplier=5, seed=None):
    """Builds a dynamic pool from existing labels and creates synthetic variations."""
    if seed is not None:
        random.seed(seed)

    print("Scanning dataset to build dynamic replacement pool...")
    dynamic_pool = build_replacement_pool_from_labels(folder_path)

    print(f"Discovered unique tickers  : {len(dynamic_pool['ticker'])}")
    print(f"Discovered unique companies: {len(dynamic_pool['company'])}")

    if len(dynamic_pool["ticker"]) < 2 or len(dynamic_pool["company"]) < 2:
        print("Error: Not enough unique labels found to safely perform swaps.")
        return

    files = glob.glob(os.path.join(folder_path, "*.json"))
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

            output_name = os.path.join(folder_path, f"augmented_m{i}_{file_filename}")
            with open(output_name, "w", encoding="utf-8") as out_f:
                json.dump(augmented_batch, out_f, indent=2, ensure_ascii=False)

    print(
        f"Generated {total_generated} augmented samples across {multiplier} variants "
        f"({total_skipped} skipped)."
    )


if __name__ == "__main__":
    labeled_folder = "data/labeled"
    run_augmentation(labeled_folder, multiplier=5)

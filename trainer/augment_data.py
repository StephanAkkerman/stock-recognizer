import copy
import glob
import json
import os
import random


def build_replacement_pool_from_labels(folder_path):
    """
    Scans all original human-labeled JSON files to extract a dynamic pool
    of unique tickers and companies based on actual annotations.
    """
    files = glob.glob(os.path.join(folder_path, "*.json"))
    # Exclude any previously augmented files from the discovery sweep
    original_files = [f for f in files if "augmented_" not in os.path.basename(f)]

    pool = {"ticker": set(), "company": set()}

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
                if r.get("type") == "labels":
                    val = r["value"]
                    label = val["labels"][0]
                    entity_text = text[val["start"] : val["end"]].strip()

                    if label in pool and entity_text:
                        # Clean off any leading cashtags for the raw dictionary pool
                        if label == "ticker" and entity_text.startswith("$"):
                            entity_text = entity_text[1:]
                        pool[label].add(entity_text)

    # Convert sets back to sorted lists so they can be indexed and randomized
    final_pool = {k: sorted(list(v)) for k, v in pool.items()}
    return final_pool


def augment_task(task, pool):
    """Swaps labeled entities from back-to-front to preserve index integrity."""
    augmented_task = copy.deepcopy(task)
    annotation = augmented_task["annotations"][0]
    results = annotation.get("result", [])

    label_results = [r for r in results if r.get("type") == "labels"]
    label_results.sort(key=lambda x: x["value"]["start"], reverse=True)

    text = augmented_task["data"]["text"]

    for r in label_results:
        val = r["value"]
        label = val["labels"][0]

        if label not in pool or not pool[label]:
            continue

        start = val["start"]
        end = val["end"]
        original_word = text[start:end]

        # Pick a random replacement from our dynamically generated pool
        replacement = random.choice(pool[label])

        # Smoothly re-apply cashtag if the original string used one
        if original_word.startswith("$") and not replacement.startswith("$"):
            replacement = "$" + replacement

        text = text[:start] + replacement + text[end:]
        len_diff = len(replacement) - len(original_word)

        val["end"] = start + len(replacement)

        for other_r in label_results:
            other_val = other_r["value"]
            if other_val["start"] > start:
                other_val["start"] += len_diff
                other_val["end"] += len_diff

    augmented_task["data"]["text"] = text
    return augmented_task


def run_augmentation(folder_path, multiplier=5):
    """Builds a dynamic pool from existing labels and creates synthetic variations."""
    print("[cyan]Scanning dataset to build dynamic replacement pool...[/cyan]")
    dynamic_pool = build_replacement_pool_from_labels(folder_path)

    print(
        f"[bold green]Discovered unique tickers  : {len(dynamic_pool['ticker'])}[/bold green]"
    )
    print(
        f"[bold green]Discovered unique companies: {len(dynamic_pool['company'])}[/bold green]"
    )

    # Safety guard: ensure we actually found entities before trying to swap
    if len(dynamic_pool["ticker"]) < 2 or len(dynamic_pool["company"]) < 2:
        print(
            "[red]Error: Not enough unique labels found to safely perform swaps.[/red]"
        )
        return

    files = glob.glob(os.path.join(folder_path, "*.json"))
    original_files = [f for f in files if "augmented_" not in os.path.basename(f)]

    total_generated = 0

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
                    augmented_batch.append(aug_task)
                    total_generated += 1
                except Exception:
                    continue

            output_name = os.path.join(folder_path, f"augmented_m{i}_{file_filename}")
            with open(output_name, "w", encoding="utf-8") as out_f:
                json.dump(augmented_batch, out_f, indent=2, ensure_ascii=False)

    print(
        f"🎉 Success! Generated {total_generated} augmented samples across {multiplier} variants."
    )


if __name__ == "__main__":
    labeled_folder = "data/labeled"
    run_augmentation(labeled_folder, multiplier=5)

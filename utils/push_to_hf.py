import glob
import json
import os

from datasets import Dataset, Features, Sequence, Value


def load_clean_gold_dataset(folder_path):
    """
    Parses all original human-labeled JSON files, stripping Label Studio metadata
    and keeping only clean text and entities. Skips augmented files.
    """
    files = glob.glob(os.path.join(folder_path, "*.json"))
    # Strictly isolate human-labeled data; exclude automated augmentations
    original_files = [f for f in files if "augmented_" not in os.path.basename(f)]

    clean_records = []

    for file_path in original_files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                ls_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {file_path}, skipping.")
                continue

        for task in ls_data:
            # Skip empty or cancelled annotations
            if not task.get("annotations") or task["annotations"][0].get(
                "was_cancelled"
            ):
                continue

            task_id = str(task.get("id", len(clean_records)))
            text = task["data"]["text"]

            # Extract clean entity spans
            entities = []
            results = task["annotations"][0].get("result", [])

            for r in results:
                if r.get("type") == "labels":
                    val = r["value"]
                    entities.append(
                        {
                            "start": int(val["start"]),
                            "end": int(val["end"]),
                            "label": str(val["labels"][0]),
                        }
                    )

            # Sort entities by start position for neatness
            entities.sort(key=lambda x: x["start"])

            clean_records.append({"id": task_id, "text": text, "entities": entities})

    return clean_records


def push_dataset_to_hub(records, repo_id):
    """Converts records to a Hugging Face Dataset with a strict schema and pushes it."""
    # Define a clean, explicit schema for GLiNER/NER tasks
    features = Features(
        {
            "id": Value("string"),
            "text": Value("string"),
            "entities": Sequence(
                {
                    "start": Value("int32"),
                    "end": Value("int32"),
                    "label": Value("string"),
                }
            ),
        }
    )

    # Sequence({...}) expects columnar format per sample: {"start": [...], "end": [...], "label": [...]}
    hf_data = {
        "id": [r["id"] for r in records],
        "text": [r["text"] for r in records],
        "entities": [
            {
                "start": [e["start"] for e in r["entities"]],
                "end": [e["end"] for e in r["entities"]],
                "label": [e["label"] for e in r["entities"]],
            }
            for r in records
        ],
    }

    print(f"📦 Creating Hugging Face Dataset with {len(records)} samples...")
    dataset = Dataset.from_dict(hf_data, features=features)

    print(f"🚀 Pushing to Hugging Face Hub: https://huggingface.co/datasets/{repo_id}")
    # This automatically handles repository creation if it doesn't exist yet
    dataset.push_to_hub(repo_id, private=True)
    print("🎉 Dataset successfully published and set to Private!")


if __name__ == "__main__":
    # --- CONFIGURATION ---
    LABELED_FOLDER = "data/labeled"
    # Replace with your HF username and desired dataset name
    HF_REPO_ID = "StephanAkkerman/wallstreetbets-ner"
    # ---------------------

    # 1. Parse and tidy up data
    gold_records = load_clean_gold_dataset(LABELED_FOLDER)

    if not gold_records:
        print("❌ No valid human-labeled records found to upload.")
    else:
        # 2. Upload to Hugging Face
        push_dataset_to_hub(gold_records, HF_REPO_ID)

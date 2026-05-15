import glob
import html
import json
import os
import re

import pandas as pd

from stock_recognizer.engine import StockRecognizer


def get_latest_adapter_path(models_dir="./models"):
    """Scans the models directory and returns the path to the highest version adapter."""
    if not os.path.exists(models_dir):
        return None

    # Find all adapter folders (e.g., reddit_adapter, reddit_adapter_v2)
    adapters = glob.glob(os.path.join(models_dir, "reddit_adapter*"))

    if not adapters:
        return None

    def extract_version(folder_path):
        folder_name = os.path.basename(folder_path)
        # If it has a '_v', extract the number
        if "_v" in folder_name:
            try:
                return int(folder_name.split("_v")[-1])
            except ValueError:
                return 1
        # If no '_v' (the original v1), treat as version 1
        return 1

    # Sort the folders descending based on their extracted version number
    adapters.sort(key=extract_version, reverse=True)

    # The first item is now the highest version
    latest_folder = adapters[0]
    final_path = os.path.join(latest_folder, "final")

    if os.path.exists(final_path):
        print(f"[Info] Found latest adapter: {final_path}")
        return final_path

    return None


class DatasetBuilder:
    def __init__(self, recognizer):
        self.recognizer = recognizer

    def create_example(self, text: str):
        entities = []
        text_upper = text.upper()

        # 1. Regex: Cashtags
        for match in self.recognizer.cashtag_re.finditer(text_upper):
            entities.append(
                {"start": match.start(), "end": match.end(), "label": "ticker"}
            )

        # 2. Regex: Company Names
        sorted_companies = sorted(
            self.recognizer.company_to_ticker.keys(), key=len, reverse=True
        )
        for company in sorted_companies:
            if len(company) < 4:
                continue
            pattern = re.compile(rf"\b{re.escape(company)}\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                if any(e["start"] <= match.start() < e["end"] for e in entities):
                    continue
                entities.append(
                    {"start": match.start(), "end": match.end(), "label": "company"}
                )

        # 3. Regex: Plain Text Tickers
        for match in self.recognizer.ticker_re.finditer(text_upper):
            ticker = match.group(0)
            clean_t = self.recognizer._clean_token(ticker)
            if (
                clean_t in self.recognizer.valid_tickers
                and clean_t not in self.recognizer.ambiguous
            ):
                if any(e["start"] <= match.start() < e["end"] for e in entities):
                    continue
                entities.append(
                    {"start": match.start(), "end": match.end(), "label": "ticker"}
                )

        # 4. AI: The LoRA Adapter
        if self.recognizer.use_ai:
            ai_results = self.recognizer.get_ai_entities(text)
            for ai_ent in ai_results:
                # Deduplicate: Only add if the AI found something the regex missed
                if not any(
                    e["start"] == ai_ent["start"] and e["end"] == ai_ent["end"]
                    for e in entities
                ):
                    entities.append(ai_ent)

        return {"text": text, "entities": entities}

    def to_label_studio_format(self, text, entities):
        """Converts raw entities into Label Studio 'predictions' format."""
        results = []
        for ent in entities:
            results.append(
                {
                    "from_name": "label",  # Matches the 'name' in your LS XML config
                    "to_name": "text",  # Matches the 'toName' in your LS XML config
                    "type": "labels",
                    "value": {
                        "start": ent["start"],
                        "end": ent["end"],
                        "labels": [ent["label"]],
                    },
                }
            )

        return {
            "data": {"text": text},
            "predictions": [
                {"model_version": "stock-recognizer-latest", "result": results}
            ],
        }

    def save_to_json(self, examples, filename):
        """Saves all examples as a single JSON array for Label Studio import."""
        ls_tasks = []
        for ex in examples:
            ls_task = self.to_label_studio_format(ex["text"], ex["entities"])
            ls_tasks.append(ls_task)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(ls_tasks, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # 1. Init
    latest_adapter = get_latest_adapter_path()

    # Pass the adapter path into your engine
    recognizer = StockRecognizer(use_ai=True, adapter_path=latest_adapter)
    builder = DatasetBuilder(recognizer)

    BATCH_NUM = 3
    start_idx = (BATCH_NUM - 1) * 100
    end_idx = BATCH_NUM * 100

    # 2. Get wsb data
    data = pd.read_csv("data/wsb.csv")[start_idx:end_idx]
    raw_texts = data["text"].tolist()

    dataset = []
    for text in raw_texts:
        example = builder.create_example(html.unescape(text))
        dataset.append(example)

    # 3. Save
    # Ensure the target directory exists
    os.makedirs("data/preds", exist_ok=True)
    builder.save_to_json(dataset, f"data/preds/train_base_{end_idx}.json")
    print(f"Generated {len(dataset)} examples. Time for manual review!")

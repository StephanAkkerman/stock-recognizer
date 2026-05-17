import html
import os
import re

import pandas as pd
from dataset_builder import (
    DatasetBuilder,
    get_latest_adapter_path,
)  # Assumes previous scripts are saved here

from stock_recognizer.engine import StockRecognizer


def clean_reddit_markdown(text: str) -> str:
    """Cleans raw Reddit text anomalies to optimize for NER tokenization."""
    if not text or not isinstance(text, str):
        return ""

    # Unescape HTML entities (&amp;, &#x200B;, etc.)
    text = html.unescape(text)

    # 1. Fix CSV double-double quotes (""text"" -> "text")
    text = text.replace('""', '"')

    # 2. Strip Markdown URLs but keep the anchor text: [Apple](https://apple.com) -> Apple
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # 3. Strip standalone URLs completely (prevents false positive tickers/companies inside links)
    text = re.sub(r"https?://\S+", "", text)

    # 4. Strip Reddit emote blocks (e.g., !(emote|t5_2th52|4271))
    text = re.sub(r"!\(emote\|[^)]+\)", "", text)

    # 5. Clean up redundant whitespace caused by stripping strings
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text


if __name__ == "__main__":
    # 1. Load the Auto-discovering Engine
    latest_adapter = get_latest_adapter_path("./models")
    recognizer = StockRecognizer(use_ai=True, adapter_path=latest_adapter)
    builder = DatasetBuilder(recognizer)

    # Set up constants for tracking
    BATCH_NUM = "final"

    # 2. Read scraped CSV data
    csv_path = "data/wallstreetbets_posts.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Please place your scraped data in {csv_path}")

    df = pd.read_csv(csv_path)

    # Process only your targeted chunk (e.g., first 100 of the new batch)
    raw_posts = df.to_dict(orient="records")

    cleaned_dataset = []

    print(
        f"Processing {len(raw_posts)} posts using {latest_adapter if latest_adapter else 'Base Model'}..."
    )

    for post in raw_posts:
        # Merge Title and Text for maximum context matching
        full_text = f"{post['title']}\n\n{post['text']}"

        # Apply the layout cleaning heuristics
        sanitized_text = clean_reddit_markdown(full_text)

        # Extract tags using combined Regex + LoRA intelligence
        example = builder.create_example(sanitized_text)
        cleaned_dataset.append(example)

    # 3. Export to JSON for easy Label Studio Import
    output_dir = "data/preds"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/batch_{BATCH_NUM}_prelabeled.json"

    builder.save_to_json(cleaned_dataset, output_file)
    print(f"🎉 Success! Generated pre-labeled file for Label Studio: {output_file}")

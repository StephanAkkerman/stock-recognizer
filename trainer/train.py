import glob
import json
import os
import re

import torch
from gliner2 import GLiNER2
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
from rich.console import Console

console = Console()


def get_next_version(models_dir="./models"):
    """Scans the models directory to determine the next version number."""
    os.makedirs(models_dir, exist_ok=True)
    adapters = glob.glob(os.path.join(models_dir, "reddit_adapter*"))

    max_v = 0
    for adapter in adapters:
        folder_name = os.path.basename(adapter)
        # Look for '_vX' at the end of the folder name
        match = re.search(r"_v(\d+)$", folder_name)
        if match:
            v = int(match.group(1))
            if v > max_v:
                max_v = v
        # Handle the original v1 folder which lacked the suffix
        elif folder_name == "reddit_adapter":
            if 1 > max_v:
                max_v = 1

    return max_v + 1


def chunk_text_with_overlap(text, chunk_word_size=150, overlap_words=40):
    """Splits text into overlapping chunks without splitting words."""
    words = text.split()
    chunks = []
    if len(words) <= chunk_word_size:
        return [text]
    step_size = chunk_word_size - overlap_words
    for i in range(0, len(words), step_size):
        chunk_text = " ".join(words[i : i + chunk_word_size])
        chunks.append(chunk_text)
        if i + chunk_word_size >= len(words):
            break
    return chunks


def parse_all_labeled_data(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.json"))
    all_clean_data = []

    descriptions = {
        "ticker": "A stock market ticker symbol, usually 1-5 letters, often preceded by a dollar sign (e.g., $AAPL, TSLA). MUST NOT be option strikes, prices, index names, or internet slang acronyms.",
        "company": "The name of a corporation, hedge fund, or business entity. MUST NOT be an uppercase ticker symbol, an index, or generic finance terms.",
    }

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            ls_data = json.load(f)

        for task in ls_data:
            if not task.get("annotations") or task["annotations"][0].get(
                "was_cancelled"
            ):
                continue

            full_text = task["data"]["text"]
            results = task["annotations"][0].get("result", [])

            # 1. Gather all Ground Truth entities for the full document
            doc_entities = {}
            for r in results:
                if r.get("type") == "labels":
                    val = r["value"]
                    label = val["labels"][0]
                    entity_text = full_text[val["start"] : val["end"]]

                    if label not in doc_entities:
                        doc_entities[label] = set()  # Use a set to deduplicate
                    doc_entities[label].add(entity_text)

            # 2. Slice the document into overlapping chunks
            chunks = chunk_text_with_overlap(
                full_text, chunk_word_size=150, overlap_words=40
            )

            # 3. Create an independent training sample for each chunk
            for chunk in chunks:
                chunk_entities_dict = {}

                for label, entity_set in doc_entities.items():
                    # Only keep the entity if it actually appears in this specific chunk
                    valid_ents = [ent for ent in entity_set if ent in chunk]
                    if valid_ents:
                        chunk_entities_dict[label] = valid_ents

                all_clean_data.append(
                    {
                        "input": chunk,
                        "output": {
                            "entities": chunk_entities_dict,
                            "entity_descriptions": descriptions,
                            "classifications": [
                                {
                                    "task": "valid",
                                    "labels": ["yes"],
                                    "true_label": ["yes"],
                                }
                            ],
                        },
                    }
                )

    return all_clean_data


if __name__ == "__main__":
    # 1. Determine next version
    next_version = get_next_version()
    adapter_name = f"reddit_adapter_v{next_version}"
    output_dir = f"./models/{adapter_name}"

    console.print(
        f"[bold cyan]Initializing Training Run for: v{next_version}[/bold cyan]"
    )

    # 2. Load the full dataset
    labeled_folder = "data/labeled"
    train_data = parse_all_labeled_data(labeled_folder)
    console.print(
        f"[bold green]Total dataset size: {len(train_data)} samples.[/bold green]"
    )

    # 3. Load Model
    base_model = GLiNER2.from_pretrained("fastino/gliner2-large-v1")
    model = torch.compile(base_model)

    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = (
        8 / BATCH_SIZE
    )  # To achieve effective batch size of 8 on limited GPU

    # 4. Training Config
    config = TrainingConfig(
        output_dir=output_dir,
        experiment_name=f"fintwit_lora_v{next_version}",
        num_epochs=25,
        batch_size=BATCH_SIZE,
        max_len=256,  # to be on the safe side with 150-word chunks + entity descriptions
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        encoder_lr=2e-5,
        task_lr=5e-4,
        use_lora=True,
        lora_r=16,
        lora_alpha=32.0,
        lora_dropout=0.1,
        lora_target_modules=["encoder"],
        save_adapter_only=True,
        fp16=False,
        bf16=True,
    )

    # 5. Train
    trainer = GLiNER2Trainer(model=model, config=config)
    trainer.train(train_data=train_data)

    console.print(
        f"[bold green]✅ v{next_version} Adapter trained and saved to {output_dir}/final/[/bold green]"
    )

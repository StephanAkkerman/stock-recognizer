import glob
import json
import os
import re

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


def parse_all_labeled_data(folder_path):
    """Finds all JSON files in the folder and parses them into GLiNER2 format."""
    files = glob.glob(os.path.join(folder_path, "*.json"))
    all_clean_data = []

    # UPDATED: Negative Prompting added to fix False Positives
    descriptions = {
        "ticker": "A stock market ticker symbol, usually 1-5 letters, often preceded by a dollar sign (e.g., $AAPL, TSLA). MUST NOT be option strikes (e.g., 140c), prices, index names, or internet slang acronyms (e.g., NFA, Lmfao, JPOW).",
        "company": "The name of a corporation, hedge fund, or business entity (e.g., Microsoft, Melvin Capital, Valve). MUST NOT be an uppercase ticker symbol, an index (e.g., Dow Jones), or generic finance terms.",
    }

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            ls_data = json.load(f)

        for task in ls_data:
            if not task.get("annotations") or task["annotations"][0].get(
                "was_cancelled"
            ):
                continue

            text = task["data"]["text"]
            entities_dict = {}
            results = task["annotations"][0].get("result", [])

            for r in results:
                if r.get("type") == "labels":
                    val = r["value"]
                    label = val["labels"][0]
                    entity_text = text[val["start"] : val["end"]]

                    if label not in entities_dict:
                        entities_dict[label] = []
                    entities_dict[label].append(entity_text)

            # Use the "Dummy Task" fix to keep negative samples
            all_clean_data.append(
                {
                    "input": text,
                    "output": {
                        "entities": entities_dict,
                        "entity_descriptions": descriptions,
                        "classifications": [
                            {"task": "valid", "labels": ["yes"], "true_label": ["yes"]}
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
    model = GLiNER2.from_pretrained("fastino/gliner2-large-v1")

    # 4. Training Config
    config = TrainingConfig(
        output_dir=output_dir,
        experiment_name=f"fintwit_lora_v{next_version}",
        num_epochs=25,
        batch_size=2,
        gradient_accumulation_steps=4,
        encoder_lr=2e-5,
        task_lr=5e-4,
        use_lora=True,
        lora_r=16,
        lora_alpha=32.0,
        lora_dropout=0.1,
        lora_target_modules=["encoder"],
        save_adapter_only=True,
        fp16=True,
    )

    # 5. Train
    trainer = GLiNER2Trainer(model=model, config=config)
    trainer.train(train_data=train_data)

    console.print(
        f"[bold green]✅ v{next_version} Adapter trained and saved to {output_dir}/final/[/bold green]"
    )

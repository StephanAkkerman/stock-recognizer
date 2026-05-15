import glob
import json
import os

from gliner2 import GLiNER2
from rich.console import Console

console = Console()


def get_errors():
    # 1. Load your V2 model
    console.print("Loading model...")
    model = GLiNER2.from_pretrained("fastino/gliner2-large-v1", local_files_only=True)
    model.load_adapter("./models/reddit_adapter_v2/final")

    # 2. Load your data
    files = glob.glob(os.path.join("data/labeled", "*.json"))

    labels_to_pass = {
        "ticker": "A stock market ticker symbol...",
        "company": "The full or partial name of a corporation...",
    }

    error_count = 0

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            ls_data = json.load(f)

        for task in ls_data:
            if not task.get("annotations") or task["annotations"][0].get(
                "was_cancelled"
            ):
                continue

            text = task["data"]["text"]

            # Get Ground Truth (What you labeled)
            gold_entities = set()
            for r in task["annotations"][0].get("result", []):
                if r.get("type") == "labels":
                    gold_entities.add(
                        (
                            text[r["value"]["start"] : r["value"]["end"]],
                            r["value"]["labels"][0],
                        )
                    )

            # Get Predictions (What the model thinks)
            raw_output = model.extract_entities(
                text, labels_to_pass, threshold=0.55, include_spans=True
            )

            pred_entities = set()
            if isinstance(raw_output, dict) and "entities" in raw_output:
                for label, items in raw_output["entities"].items():
                    for item in items:
                        pred_entities.add((item["text"], label))

            # Compare
            false_positives = (
                pred_entities - gold_entities
            )  # Model hallucinated or got boundaries wrong
            false_negatives = (
                gold_entities - pred_entities
            )  # Model completely missed it

            if false_positives or false_negatives:
                error_count += 1
                console.print(f"\n[bold white]Text:[/bold white] {text}")
                if false_negatives:
                    console.print(
                        f"[bold red]Missed (FN):[/bold red] {false_negatives}"
                    )
                if false_positives:
                    console.print(
                        f"[bold yellow]Hallucinated/Mismatched (FP):[/bold yellow] {false_positives}"
                    )

                if (
                    error_count >= 15
                ):  # Just show the first 15 so we don't spam your console
                    return


if __name__ == "__main__":
    get_errors()

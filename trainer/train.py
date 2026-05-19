import glob
import json
import os
import random
import re

import numpy as np
import torch
from gliner2 import GLiNER2
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
from rich.console import Console

console = Console()

SEED = 42
VAL_FRACTION = 0.15

ENTITY_DESCRIPTIONS = {
    "ticker": "A stock market ticker symbol, usually 1-5 letters, often preceded by a dollar sign (e.g., $AAPL, TSLA). MUST NOT be option strikes, prices, index names, or internet slang acronyms.",
    "company": "The name of a corporation, hedge fund, or business entity. MUST NOT be an uppercase ticker symbol, an index, or generic finance terms.",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_next_version(models_dir="./models"):
    """Scans the models directory to determine the next version number."""
    os.makedirs(models_dir, exist_ok=True)
    adapters = glob.glob(os.path.join(models_dir, "reddit_adapter*"))

    max_v = 0
    for adapter in adapters:
        folder_name = os.path.basename(adapter)
        match = re.search(r"_v(\d+)$", folder_name)
        if match:
            v = int(match.group(1))
            if v > max_v:
                max_v = v
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


def entity_in_chunk(entity_text, chunk):
    """Token-aware containment check: matches whole tokens only, $-prefixed tickers included."""
    return re.search(rf"(?<!\w){re.escape(entity_text)}(?!\w)", chunk) is not None


def task_to_samples(task):
    """Converts one Label Studio task into one or more chunked training samples."""
    full_text = task["data"]["text"]
    results = task["annotations"][0].get("result", [])

    doc_entities = {}
    for r in results:
        if r.get("type") != "labels":
            continue
        val = r["value"]
        label = val["labels"][0]
        entity_text = full_text[val["start"] : val["end"]]
        doc_entities.setdefault(label, set()).add(entity_text)

    chunks = chunk_text_with_overlap(full_text, chunk_word_size=150, overlap_words=40)

    samples = []
    for chunk in chunks:
        chunk_entities_dict = {}
        for label, entity_set in doc_entities.items():
            valid_ents = [ent for ent in entity_set if entity_in_chunk(ent, chunk)]
            if valid_ents:
                chunk_entities_dict[label] = valid_ents

        samples.append(
            {
                "input": chunk,
                "output": {
                    "entities": chunk_entities_dict,
                    "entity_descriptions": ENTITY_DESCRIPTIONS,
                    # Keeps empty-entity chunks from being dropped by the trainer
                    # so the model sees real negatives. See discussion of the
                    # `valid: yes` degeneracy in the train.py review.
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
    return samples


def _load_tasks(folder_path):
    """Yields (task, source_file) for every usable annotation in a folder."""
    if not folder_path or not os.path.isdir(folder_path):
        return
    for fp in glob.glob(os.path.join(folder_path, "*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            ls_data = json.load(f)
        for task in ls_data:
            if not task.get("annotations") or task["annotations"][0].get(
                "was_cancelled"
            ):
                continue
            yield task, fp


def parse_all_labeled_data(
    labeled_folder,
    augmented_folder=None,
    test_folder=None,
    val_fraction=VAL_FRACTION,
    seed=SEED,
):
    """Loads originals from `labeled_folder` and (optionally) augmented variants
    from `augmented_folder`, splitting by source task id so augmented copies of
    validation tasks never leak into training.

    If `test_folder` is given, every task id present there is excluded from
    train and val — including augmented duplicates that carry the same id —
    so the held-out test set never contaminates training.
    """
    test_ids = set()
    if test_folder and os.path.isdir(test_folder):
        for task, _ in _load_tasks(test_folder):
            test_ids.add(task["id"])

    originals = [(t, fp) for t, fp in _load_tasks(labeled_folder) if t["id"] not in test_ids]

    # Split task ids deterministically.
    rng = random.Random(seed)
    ids = sorted({task["id"] for task, _ in originals})
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_fraction))
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])

    train_samples, val_samples = [], []

    for task, _ in originals:
        tid = task["id"]
        if tid in val_ids:
            val_samples.extend(task_to_samples(task))
        elif tid in train_ids:
            train_samples.extend(task_to_samples(task))

    # Augmented: training only, and only for tasks whose source is in train_ids.
    # The `test_ids` guard catches augmented variants of test tasks that still
    # live on disk (e.g. `augmented_m*_labeled_300.json` after the 300-file move).
    for task, _ in _load_tasks(augmented_folder):
        tid = task["id"]
        if tid in test_ids:
            continue
        if tid in train_ids:
            train_samples.extend(task_to_samples(task))

    return train_samples, val_samples


if __name__ == "__main__":
    set_seed(SEED)

    next_version = get_next_version()
    adapter_name = f"reddit_adapter_v{next_version}"
    output_dir = f"./models/{adapter_name}"

    console.print(
        f"[bold cyan]Initializing Training Run for: v{next_version}[/bold cyan]"
    )

    labeled_folder = "data/labeled"
    augmented_folder = "data/augmented"
    test_folder = "data/test"
    train_data, val_data = parse_all_labeled_data(
        labeled_folder, augmented_folder, test_folder=test_folder
    )
    console.print(
        f"[bold green]Train: {len(train_data)} samples | Val: {len(val_data)} samples[/bold green]"
    )

    base_model = GLiNER2.from_pretrained("fastino/gliner2-large-v1")
    model = torch.compile(base_model)

    BATCH_SIZE = 4
    EFFECTIVE_BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // BATCH_SIZE

    config = TrainingConfig(
        output_dir=output_dir,
        experiment_name=f"fintwit_lora_v{next_version}",
        num_epochs=10,
        batch_size=BATCH_SIZE,
        max_len=256,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        encoder_lr=2e-5,
        task_lr=5e-4,
        max_grad_norm=1.0,
        use_lora=True,
        lora_r=16,
        lora_alpha=32.0,
        lora_dropout=0.1,
        lora_target_modules=["encoder"],
        save_adapter_only=True,
        fp16=False,
        bf16=True,
        seed=SEED,
        # v5 collapsed to "predict nothing" because we 5x'd the data via
        # augmentation but kept num_epochs=25 and never acted on validation.
        # eval_strategy="steps" + eval_steps=500 are the defaults; early
        # stopping uses them to bail when val loss stops improving.
        early_stopping=True,
        early_stopping_patience=3,
    )

    trainer = GLiNER2Trainer(model=model, config=config)
    trainer.train(train_data=train_data, eval_data=val_data)

    console.print(
        f"[bold green]v{next_version} Adapter trained and saved to {output_dir}/final/[/bold green]"
    )

    # Benchmark the freshly trained adapter and persist the result so the
    # next ``python trainer/benchmark.py`` run can short-circuit this version.
    try:
        from trainer.benchmark import benchmark_adapter, locate_adapter_weights
    except ImportError:
        from benchmark import benchmark_adapter, locate_adapter_weights

    adapter_final = locate_adapter_weights(output_dir)
    if adapter_final is None:
        console.print(
            f"[yellow]No adapter weights found under {output_dir}; "
            "skipping post-train benchmark.[/yellow]"
        )
        raise SystemExit(0)
    adapter_label = f"GLiNER2 Large + Adapter v{next_version}"
    training_params = {
        "num_epochs": config.num_epochs,
        "batch_size": config.batch_size,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "encoder_lr": config.encoder_lr,
        "task_lr": config.task_lr,
        "max_grad_norm": config.max_grad_norm,
        "early_stopping": config.early_stopping,
        "early_stopping_patience": config.early_stopping_patience,
        "seed": SEED,
        "val_fraction": VAL_FRACTION,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
    }
    console.print(f"[cyan]Benchmarking {adapter_label}...[/cyan]")
    metrics, test_hash = benchmark_adapter(
        adapter_label, adapter_final, training_params=training_params
    )
    overall = metrics["overall"]
    console.print(
        f"[bold]{adapter_label}[/bold] vs test set [yellow]{test_hash}[/yellow]: "
        f"P={overall['p']:.2%}  R={overall['r']:.2%}  F1={overall['f1']:.2%}"
    )

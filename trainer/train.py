import glob
import hashlib
import json
import os
import random
import re
import subprocess
from datetime import datetime, timezone

import numpy as np
import torch
from gliner2 import GLiNER2
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
from rich.console import Console

console = Console()

SEED = 42
EARLY_STOPPING = True
EPOCHS = 10
VAL_FRACTION = 0.15

ENTITY_DESCRIPTIONS = {
    "ticker": (
        "An all-uppercase abbreviation (1–6 letters, with or without a leading $) "
        "that refers to a tradeable security. Examples: $AAPL, TSLA, GME, META, GOOGL, SPY. "
        "Label as ticker even when the abbreviation also names the company — GME, COST, "
        "META, GOOGL are tickers, not companies, even in earnings or thesis contexts. "
        "MUST NOT be option strikes (e.g. 140c), dollar amounts, index names spelled out, "
        "or internet slang (e.g. NFA, YOLO, JPOW)."
    ),
    "company": (
        "The full or informal name of a company written in mixed or natural case — "
        "e.g. 'Apple', 'GameStop', 'Rocket Lab', 'Bed Bath & Beyond', 'upstart'. "
        "MUST NOT be an all-uppercase abbreviation — those are tickers. "
        "MUST NOT be a generic finance term, index name, or person's name."
    ),
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

    originals = [
        (t, fp) for t, fp in _load_tasks(labeled_folder) if t["id"] not in test_ids
    ]

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


def _git_commit():
    """Best-effort: return current git HEAD, or None if not in a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=5
        )
        return out.decode().strip()
    except Exception:
        return None


def _summarise_labeled_file(fp):
    """Read a Label Studio export, return tasks/entities/sha1 summary."""
    with open(fp, "rb") as f:
        raw = f.read()
    sha1 = hashlib.sha1(raw).hexdigest()[:12]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    n_tasks = 0
    n_ents = 0
    for t in data if isinstance(data, list) else []:
        if not t.get("annotations"):
            continue
        if t["annotations"][0].get("was_cancelled"):
            continue
        n_tasks += 1
        n_ents += sum(
            1 for r in t["annotations"][0].get("result", []) if r.get("type") == "labels"
        )
    return {"name": os.path.basename(fp), "tasks": n_tasks, "entities": n_ents, "sha1": sha1}


def gather_training_metadata(
    config,
    train_count,
    val_count,
    effective_batch_size,
    labeled_folder,
    test_folder,
    augmented_folder,
):
    """Snapshot everything needed to reproduce this training run.

    Written next to the adapter as `training_metadata.json` so benchmark.py
    can later merge it into the per-adapter params it persists — keeping
    full training context attached to each version even when re-benchmarked
    months later.
    """
    labeled_files = []
    for fp in sorted(glob.glob(os.path.join(labeled_folder, "*.json"))):
        name = os.path.basename(fp)
        if name.endswith(".bak"):
            continue
        if "negatives" in name:
            continue
        info = _summarise_labeled_file(fp)
        if info:
            labeled_files.append(info)

    test_files = []
    test_held_out = 0
    if os.path.isdir(test_folder):
        for fp in sorted(glob.glob(os.path.join(test_folder, "*.json"))):
            info = _summarise_labeled_file(fp)
            if info:
                test_files.append(info)
                test_held_out += info["tasks"]

    # Pull augmentation knobs at training time. These are module-level
    # constants in utils.augment_data; if the user mutated them between
    # `python utils/augment_data.py` and `python trainer/train.py`, the
    # captured values reflect the latter — best-effort, not authoritative.
    aug = {}
    try:
        from utils.augment_data import CASHTAG_FORMAT_PROB, EXPANDED_POOL_WEIGHT

        aug["expanded_pool_weight"] = EXPANDED_POOL_WEIGHT
        aug["cashtag_format_prob"] = CASHTAG_FORMAT_PROB
    except ImportError:
        pass
    aug["augmented_files"] = (
        len(glob.glob(os.path.join(augmented_folder, "*.json")))
        if augmented_folder and os.path.isdir(augmented_folder)
        else 0
    )

    negatives = None
    for fp in sorted(glob.glob(os.path.join(labeled_folder, "negatives*.json"))):
        info = _summarise_labeled_file(fp)
        if info:
            negatives = info
            break

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "seed": SEED,
        "config": {
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "effective_batch_size": effective_batch_size,
            "encoder_lr": config.encoder_lr,
            "task_lr": config.task_lr,
            "max_grad_norm": config.max_grad_norm,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "early_stopping": config.early_stopping,
            "early_stopping_patience": getattr(config, "early_stopping_patience", None),
            "val_fraction": VAL_FRACTION,
        },
        "data": {
            "labeled_files": labeled_files,
            "test_files": test_files,
            "test_held_out_tasks": test_held_out,
            "train_samples": train_count,
            "val_samples": val_count,
        },
        "augmentation": aug,
        "negatives": negatives,
    }


if __name__ == "__main__":
    set_seed(SEED)

    # Refresh the held-out test split before loading training data so any
    # newly added labeled_*.json gets its ~15% stratified slice held out
    # automatically. Deterministic via the same SEED, so the split is
    # reproducible across runs as long as the source files don't change.
    try:
        from trainer.split_test_set import run as refresh_test_split
    except ImportError:
        from split_test_set import run as refresh_test_split

    console.print("[bold cyan]Refreshing stratified test split...[/bold cyan]")
    refresh_test_split(seed=SEED)

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
        num_epochs=EPOCHS,
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
        # Early stopping IS necessary here. v9 (25 epochs, no early stop, w/
        # 55 hard negatives) collapsed to "predict nothing" by step ~6000 —
        # all intermediate checkpoints from 6000 onward score 0% F1. The
        # negatives dominate the gradient once training runs long enough.
        # Patience=3 stops on the noisy val signal, but stopping early is
        # better than collapse.
        early_stopping=EARLY_STOPPING,
        early_stopping_patience=3,
    )

    trainer = GLiNER2Trainer(model=model, config=config)
    trainer.train(train_data=train_data, eval_data=val_data)

    console.print(
        f"[bold green]v{next_version} Adapter trained and saved to {output_dir}/final/[/bold green]"
    )

    # Snapshot the full training context next to the adapter so re-benchmark
    # runs months later still know which data + config produced this version.
    metadata = gather_training_metadata(
        config,
        train_count=len(train_data),
        val_count=len(val_data),
        effective_batch_size=EFFECTIVE_BATCH_SIZE,
        labeled_folder=labeled_folder,
        test_folder=test_folder,
        augmented_folder=augmented_folder,
    )
    metadata_path = os.path.join(output_dir, "training_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    console.print(f"[cyan]Wrote training metadata to {metadata_path}[/cyan]")

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
    console.print(f"[cyan]Benchmarking {adapter_label}...[/cyan]")
    metrics, test_hash = benchmark_adapter(adapter_label, adapter_final)
    overall = metrics["overall"]
    console.print(
        f"[bold]{adapter_label}[/bold] vs test set [yellow]{test_hash}[/yellow]: "
        f"P={overall['p']:.2%}  R={overall['r']:.2%}  F1={overall['f1']:.2%}"
    )

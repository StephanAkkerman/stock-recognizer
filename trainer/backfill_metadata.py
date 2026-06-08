"""One-shot backfill of `training_metadata.json` for pre-v14 adapters.

For each existing adapter folder this script:
  1. Reads the trainer-written `training_config.json` (always present —
     authoritative for hyperparameters).
  2. Looks up hand-encoded session-history facts for data composition,
     augmentation, and negatives — clearly marked as `backfilled: true`.
  3. Uses the adapter weight file's mtime as a `timestamp` proxy.
  4. Writes `training_metadata.json` next to the adapter.

Re-run `python trainer/benchmark.py` afterward — its `_load_training_metadata`
path will pick up the new files and persist the params into the store.

This is a one-shot. v14+ adapters get the same file written automatically by
train.py with full git/timestamp/SHA accuracy. Don't rely on this script for
new adapters.
"""

import json
import os
from datetime import datetime, timezone

from rich.console import Console

console = Console()

MODELS_DIR = "./models"

# Session-history facts. Encode only what I'm confident about; leave unknowns
# explicit. Each entry's "confidence" annotates how reliable the data is.
#
# Keys per version:
#   data: which labeled files were in data/labeled at training time, and
#         which negatives file (if any) was active. Use named sets:
#           'labeled_4_files'  = labeled_{100,200,300,final}.json (pre-test-split)
#           'labeled_3_files'  = labeled_{100,200,final}.json     (labeled_300 in test/)
#           'labeled_4_strat'  = same as 4_files but only ~85% trained (stratified split)
#           'labeled_5_files'  = adds labeled_auto_labeled.json   (post-auto-label)
#   augmentation: multiplier + which knobs were active
#   negatives: bool — was 55-task negatives_mined.json in labeled/?
HISTORICAL_FACTS = {
    1: {
        "data_state": "labeled_4_files",
        "augmentation": {
            "multiplier": None,
            "expanded_pool": False,
            "expanded_pool_weight": 0.0,
            "cashtag_format_prob": 0.0,
        },
        "negatives_count": 0,
        "note": "Pre-session adapter. r=8/alpha=16 (older LoRA config). "
        "Augmentation details unknown — multiplier not recoverable.",
    },
    2: {
        "data_state": "labeled_4_files",
        "augmentation": {
            "multiplier": None,
            "expanded_pool": False,
            "expanded_pool_weight": 0.0,
            "cashtag_format_prob": 0.0,
        },
        "negatives_count": 0,
        "note": "Pre-session. First r=16/alpha=32 run. Multiplier not recoverable.",
    },
    3: {
        "data_state": "labeled_4_files",
        "augmentation": {
            "multiplier": None,
            "expanded_pool": False,
            "expanded_pool_weight": 0.0,
            "cashtag_format_prob": 0.0,
        },
        "negatives_count": 0,
        "note": "Pre-session.",
    },
    4: {
        "data_state": "labeled_4_files",
        "augmentation": {
            "multiplier": None,
            "expanded_pool": False,
            "expanded_pool_weight": 0.0,
            "cashtag_format_prob": 0.0,
        },
        "negatives_count": 0,
        "note": "Pre-session. Best adapter on old test set (saw labeled_300 in training, "
        "which became part of the original test set — contamination known).",
    },
    5: {
        "data_state": "labeled_4_files",
        "augmentation": {
            "multiplier": 5,
            "expanded_pool": False,
            "expanded_pool_weight": 0.0,
            "cashtag_format_prob": 0.0,
        },
        "negatives_count": 0,
        "note": "Pre-session. 5x labeled augmentation; first run with early stopping.",
    },
    6: {
        "data_state": "labeled_3_files",
        "augmentation": {
            "multiplier": 3,
            "expanded_pool": True,
            "expanded_pool_weight": 0.7,
            "cashtag_format_prob": 0.0,
        },
        "negatives_count": 0,
        "note": "First with expanded-pool augmentation from financedatabase. "
        "Confirmed harmful — company-name poisoning (Beyond/Flex/Olympic).",
    },
    7: {
        "data_state": "labeled_3_files",
        "augmentation": {
            "multiplier": 3,
            "expanded_pool": True,
            "expanded_pool_weight": 0.7,
            "cashtag_format_prob": 0.0,
        },
        "negatives_count": 55,
        "note": "v6 setup + 55 hard negatives. Confounded by v6's expanded-pool poison.",
    },
    8: {
        "data_state": "labeled_3_files",
        "augmentation": {
            "multiplier": 3,
            "expanded_pool": False,
            "expanded_pool_weight": 0.0,
            "cashtag_format_prob": 0.0,
        },
        "negatives_count": 55,
        "note": "Clean labeled-only aug + 55 negatives. First negatives-only test.",
    },
    9: {
        "data_state": "labeled_3_files",
        "augmentation": {
            "multiplier": 3,
            "expanded_pool": False,
            "expanded_pool_weight": 0.0,
            "cashtag_format_prob": 0.0,
        },
        "negatives_count": 55,
        "note": "v8 data with 25 epochs no early stopping. final/ checkpoint collapsed "
        "to 0% F1; best/val_loss checkpoint scored 65-72% depending on test set. "
        "lora_b_norm 7.85 = very early checkpoint.",
    },
    10: {
        "data_state": "labeled_3_files",
        "augmentation": {
            "multiplier": 3,
            "expanded_pool": False,
            "expanded_pool_weight": 0.0,
            "cashtag_format_prob": 0.30,
        },
        "negatives_count": 55,
        "note": "First run with cashtag-format augmentation. Strong on stratified test.",
    },
    11: {
        "data_state": "labeled_3_files",
        "augmentation": {
            "multiplier": 3,
            "expanded_pool": False,
            "expanded_pool_weight": 0.0,
            "cashtag_format_prob": 0.30,
        },
        "negatives_count": 0,  # negatives moved to .bak before v11
        "note": "v10 setup minus negatives, with 25 epochs no early stopping. "
        "lora_b_norm 27.12 = overtrained.",
    },
    12: {
        "data_state": "labeled_4_strat",
        "augmentation": {
            "multiplier": 5,
            "expanded_pool": False,
            "expanded_pool_weight": 0.0,
            "cashtag_format_prob": 0.30,
        },
        "negatives_count": 0,
        "note": "First adapter trained after stratified-split restored labeled_300. "
        "5x multiplier + cashtag aug, no negatives.",
    },
    13: {
        "data_state": "labeled_5_files",
        "augmentation": {
            "multiplier": 3,
            "expanded_pool": False,
            "expanded_pool_weight": 0.0,
            "cashtag_format_prob": 0.30,
        },
        "negatives_count": 0,
        "note": "First adapter with LLM-auto-labeled data included. Topic-distribution "
        "mismatch with old meme-stock-heavy test partially explained regression.",
    },
}


# Snapshot of file inventories by data_state. Captured from disk on
# 2026-06-08 — `tasks` and `entities` counts include the valid annotated
# rows (was_cancelled excluded). sha1s are 12-char prefixes of the file
# bytes at that point — they won't match historical file state perfectly
# but give us a reproducibility anchor for "what's in this file now".
def _summarise_file(fp):
    import hashlib

    if not os.path.exists(fp):
        return None
    with open(fp, "rb") as f:
        raw = f.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    n_tasks, n_ents = 0, 0
    for t in data if isinstance(data, list) else []:
        if not t.get("annotations"):
            continue
        if t["annotations"][0].get("was_cancelled"):
            continue
        n_tasks += 1
        n_ents += sum(
            1
            for r in t["annotations"][0].get("result", [])
            if r.get("type") == "labels"
        )
    return {
        "name": os.path.basename(fp),
        "tasks": n_tasks,
        "entities": n_ents,
        "sha1": hashlib.sha1(raw).hexdigest()[:12],
    }


DATA_STATE_FILES = {
    "labeled_4_files": [
        "data/labeled/labeled_100.json",
        "data/labeled/labeled_200.json",
        "data/labeled/labeled_300.json",
        "data/labeled/labeled_final.json",
    ],
    "labeled_3_files": [
        "data/labeled/labeled_100.json",
        "data/labeled/labeled_200.json",
        "data/labeled/labeled_final.json",
    ],
    "labeled_4_strat": [
        "data/labeled/labeled_100.json",
        "data/labeled/labeled_200.json",
        "data/labeled/labeled_300.json",
        "data/labeled/labeled_final.json",
    ],
    "labeled_5_files": [
        "data/labeled/labeled_100.json",
        "data/labeled/labeled_200.json",
        "data/labeled/labeled_300.json",
        "data/labeled/labeled_final.json",
        "data/labeled/labeled_auto_labeled.json",
    ],
}


def _adapter_dir(version):
    if version == 1:
        # v1 was named without suffix originally
        for cand in (f"{MODELS_DIR}/reddit_adapter_v1", f"{MODELS_DIR}/reddit_adapter"):
            if os.path.isdir(cand):
                return cand
        return None
    cand = f"{MODELS_DIR}/reddit_adapter_v{version}"
    return cand if os.path.isdir(cand) else None


def _adapter_timestamp(adapter_dir):
    """Use the adapter weights file mtime as a proxy for training completion."""
    for sub in ("best", "final"):
        weight = os.path.join(adapter_dir, sub, "adapter_model.safetensors")
        if os.path.exists(weight):
            return datetime.fromtimestamp(
                os.path.getmtime(weight), tz=timezone.utc
            ).isoformat()
    return None


def build_metadata_for_version(version, facts, adapter_dir):
    """Construct a training_metadata.json payload for one adapter version."""
    cfg_path = os.path.join(adapter_dir, "training_config.json")
    if not os.path.exists(cfg_path):
        console.print(
            f"[yellow]v{version}: no training_config.json — skipping.[/yellow]"
        )
        return None
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    labeled_files = []
    for fp in DATA_STATE_FILES.get(facts["data_state"], []):
        info = _summarise_file(fp)
        if info:
            labeled_files.append(info)

    return {
        "backfilled": True,
        "backfill_note": facts["note"],
        "timestamp": _adapter_timestamp(adapter_dir),
        "git_commit": None,
        "seed": cfg.get("seed", 42),
        "config": {
            "num_epochs": cfg.get("num_epochs"),
            "batch_size": cfg.get("batch_size"),
            "effective_batch_size": (
                cfg.get("batch_size", 0) * cfg.get("gradient_accumulation_steps", 1)
            ),
            "encoder_lr": cfg.get("encoder_lr"),
            "task_lr": cfg.get("task_lr"),
            "max_grad_norm": cfg.get("max_grad_norm"),
            "lora_r": cfg.get("lora_r"),
            "lora_alpha": cfg.get("lora_alpha"),
            "lora_dropout": cfg.get("lora_dropout"),
            "early_stopping": cfg.get("early_stopping"),
            "early_stopping_patience": cfg.get("early_stopping_patience"),
            "val_fraction": 0.15 if version >= 6 else None,  # unknown for v1-v5
        },
        "data": {
            "labeled_files": labeled_files,
            "data_state": facts["data_state"],
            "train_samples": None,  # not recoverable for backfilled rows
            "val_samples": None,
        },
        "augmentation": {
            "multiplier": facts["augmentation"]["multiplier"],
            "expanded_pool": facts["augmentation"]["expanded_pool"],
            "expanded_pool_weight": facts["augmentation"]["expanded_pool_weight"],
            "cashtag_format_prob": facts["augmentation"]["cashtag_format_prob"],
            "augmented_files": None,  # disk state has churned since these runs
        },
        "negatives": (
            {"name": "negatives_mined.json", "tasks": facts["negatives_count"]}
            if facts["negatives_count"] > 0
            else None
        ),
    }


def main():
    written = []
    skipped = []
    for version, facts in HISTORICAL_FACTS.items():
        adapter_dir = _adapter_dir(version)
        if not adapter_dir:
            skipped.append((version, "adapter folder missing"))
            continue
        out_path = os.path.join(adapter_dir, "training_metadata.json")
        if os.path.exists(out_path):
            skipped.append((version, f"already exists at {out_path}"))
            continue
        metadata = build_metadata_for_version(version, facts, adapter_dir)
        if metadata is None:
            skipped.append((version, "no training_config.json"))
            continue
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        written.append((version, out_path))

    console.print(f"[bold green]Backfilled {len(written)} adapter(s)[/bold green]")
    for v, path in written:
        console.print(f"  v{v}: {path}")
    if skipped:
        console.print(f"\n[yellow]Skipped {len(skipped)}:[/yellow]")
        for v, reason in skipped:
            console.print(f"  v{v}: {reason}")

    console.print(
        "\n[cyan]Next:[/cyan] run `python trainer/benchmark.py` so the new "
        "params land in models/benchmark_results.json."
    )


if __name__ == "__main__":
    main()

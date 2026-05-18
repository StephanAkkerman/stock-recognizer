"""Persistent benchmark-result cache.

Results are keyed by ``(adapter_name, test_set_hash)`` so re-running the
benchmark with the same labeled data short-circuits — only newly added
adapters (or a changed test set) need fresh inference.
"""

import hashlib
import json
import os
from datetime import datetime, timezone

from safetensors.torch import load_file

STORE_PATH = os.path.join("models", "benchmark_results.json")


def compute_test_set_hash(dataset):
    """Stable, order-independent fingerprint of a parsed test set.

    ``dataset`` is the list returned by ``parse_all_label_studio_exports`` —
    each entry is ``{"text": str, "entities": [{"start", "end", "label"}, ...]}``.
    Any change to text or annotations changes the hash and invalidates the
    cached metrics for that test set.
    """
    canonical = []
    for entry in dataset:
        ents = sorted(
            (e["start"], e["end"], e["label"]) for e in entry["entities"]
        )
        canonical.append([entry["text"], ents])
    canonical.sort()
    blob = json.dumps(canonical, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def load_store(path=STORE_PATH):
    if not os.path.exists(path):
        return {"test_sets": {}, "results": {}}
    with open(path, "r", encoding="utf-8") as f:
        store = json.load(f)
    store.setdefault("test_sets", {})
    store.setdefault("results", {})
    return store


def save_store(store, path=STORE_PATH):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)


def register_test_set(store, hash_, dataset, source_folder=None):
    if hash_ in store["test_sets"]:
        return
    store["test_sets"][hash_] = {
        "num_tasks": len(dataset),
        "num_entities": sum(len(e["entities"]) for e in dataset),
        "source_folder": source_folder,
        "first_seen": _utc_now(),
    }


def get_cached(store, model_name, hash_):
    return store["results"].get(model_name, {}).get(hash_)


def put_result(store, model_name, hash_, metrics, params=None):
    store["results"].setdefault(model_name, {})[hash_] = {
        "metrics": metrics,
        "params": params or {},
        "evaluated_at": _utc_now(),
    }


def derive_adapter_params(adapter_path):
    """Extract intrinsic, weight-derived facts about an adapter on disk.

    These are the signals that don't require knowing the training command —
    file size, the lora_B Frobenius norm (a proxy for how aggressively the
    adapter is overriding the base model), and the PEFT config values that
    can drift across training runs.
    """
    out = {}
    weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if os.path.exists(weights_path):
        out["adapter_size_kb"] = round(os.path.getsize(weights_path) / 1024, 1)
        weights = load_file(weights_path)
        sq = 0.0
        for k, v in weights.items():
            if "lora_B" in k:
                sq += v.float().pow(2).sum().item()
        out["lora_b_norm"] = round(sq ** 0.5, 4)
    cfg_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for k in ("r", "lora_alpha", "lora_dropout", "target_modules"):
            if k in cfg:
                out[k] = cfg[k]
    return out

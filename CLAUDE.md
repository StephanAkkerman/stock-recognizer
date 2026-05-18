# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (GPU training requires torch with CUDA separately)
pip install -e .
pip install -r requirements.txt

# Run all tests
pytest

# Run a single test by name
pytest tests/test_engine.py::test_cashtag_basic -v

# Lint
ruff check .

# Format
black .
```

## Architecture

**`StockRecognizer`** (`stock_recognizer/engine.py`) exposes two public methods:
- `recognize(text)` — regex-only, fast, no model required
- `recognize_ai(text)` — regex + GLiNER2 AI model, higher recall

Both return a list of uppercase ticker strings. Recognition is stateless after construction; all state lives in the instance (compiled regexes, market data dicts, optional loaded model).

**Recognition pipeline (regex path)**:
1. Cashtags (`$AAPL`) — trusted unconditionally, no market validation needed
2. Plain uppercase tokens (2–6 letters) matched against `valid_tickers` (built from `financedatabase`)
3. Filtered through `AMBIGUOUS_WORDS` (498-word blacklist in `constants.py`) and exchange suffixes

**Recognition pipeline (AI path)**:
Runs regex first, then GLiNER2 extracts `ticker` and `company` entities. Company names are resolved to tickers via the market-data mapping built at construction. Results are merged and deduplicated with the regex output.

**Two-tier entity blacklisting**: `AMBIGUOUS_WORDS` handles general words; the exchange blacklist (`EXCHANGE_BLACKLIST`) handles international suffix noise like `.SA`, `.KL`, `.F`.

## Training Pipeline

All training artifacts live outside the repo (git-ignored):
- `models/reddit_adapter_v{N}/final/` — active LoRA adapters
- Base model: `fastino/gliner2-large-v1` loaded from HuggingFace

**End-to-end data flow**:
```
scraper.py → CSV → cleaner.py → preds/ → Label Studio (manual review) → data/labeled/
                                                                                ↓
                                                            augment_data.py → data/augmented/
                                                                                ↓
                                                                          train.py → models/
```

**Train/val split** is done by original task ID (Label Studio `id` field), not by chunk. Augmented files carry the same task IDs as their source, so they are automatically routed to training only — never validation. Val fraction is 10% of unique task IDs.

**Chunking**: Documents are split into 150-word chunks with 40-word overlap. Each chunk becomes an independent training sample. Chunks with no entities are kept (they provide negative supervision via the `classifications` field).

**Key config in `train.py`**: `SEED`, `VAL_FRACTION`, `BATCH_SIZE`, `EFFECTIVE_BATCH_SIZE` are module-level constants. LoRA settings (`r=16`, `alpha=32`, `dropout=0.1`) are in `TrainingConfig`.

## Data Format

Label Studio exports JSON arrays. Each task has:
- `task["data"]["text"]` — raw post text
- `task["annotations"][0]["result"]` — list of span annotations with `value.start`, `value.end`, `value.text`, `value.labels`
- `task["id"]` — stable integer used for train/val splitting

Labels: `ticker` and `company` (defined in `data/label_studio.xml`).

Augmented files in `data/augmented/` have stale `prediction` blocks deliberately cleared (offsets no longer match the swapped text).

## Benchmarking

```bash
python trainer/benchmark.py
```

Evaluates all adapter versions found under `models/`, calculates per-entity precision/recall/F1, and prints a Rich table. Uses chunked inference matching the training chunk size.

## Code Style

Black (line-length 88), Ruff, NumPy-style docstrings. Enforced via CI.

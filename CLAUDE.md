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

## Labeling Policy: ticker vs company

The two labels serve distinct roles in the engine's resolution pipeline, so they must be applied consistently. Use this decision tree in order:

**1. Label as `ticker`** if the span text — uppercased and with `$` stripped — exactly matches a symbol in `valid_tickers`. This covers cashtags, all-caps symbols, and informally lowercased ticker references:

| Text in post | Label | Why |
|---|---|---|
| `$AMC`, `$amc` | ticker | cashtag |
| `AMC`, `META`, `NVDA` | ticker | exact ticker symbol |
| `gme`, `amc`, `tsla` | ticker | informal lowercase, resolves to ticker |

**2. Label as `company`** if the span is a company name that resolves via `company_to_ticker` but is not itself a `valid_tickers` symbol:

| Text in post | Label | Why |
|---|---|---|
| `Meta`, `Microsoft`, `Nvidia` | company | written name, ticker is META/MSFT/NVDA |
| `AMC Theatres`, `Goldman Sachs` | company | multi-word name |
| `NVIDIA`, `TSMC`, `APPLE` | company | all-caps abbreviation, ticker differs (NVDA/TSM/AAPL) |

**3. Do not label** government agencies, regulatory bodies, financial metrics, or media outlets — they cannot resolve to a ticker:

- Government/regulatory: `CSRC`, `NASA`, `SEC`, `FINRA`, `NDAA`
- Financial acronyms: `PDT`, `EV` (enterprise value), `SG&A`, `RSUs`, `PT` (price target)
- Media/private: `CNBC`, `Bloomberg`, `The Verge`

**Key rule for AMC-like entities** where the company abbreviation equals its ticker: always use the form in the text. `AMC` → ticker. `AMC Theatres` → company. The engine handles both paths to the same resolved output, so consistency matters more than semantic intent.

Run `python trainer/fix_label_policy.py` to find and fix policy violations in `data/labeled/` and `data/test/`. After fixing labeled data, regenerate augmented data.

## Benchmarking

```bash
python trainer/benchmark.py
```

Evaluates all adapter versions found under `models/`, calculates per-entity precision/recall/F1, and prints a Rich table. Uses chunked inference matching the training chunk size.

### Scoring contract: set-based, deduplicated per document

The project's goal is to know **what** a post is talking about, not how many times a symbol is repeated — an entity counts as caught if the model finds it **at least once** in the document. This mirrors the engine's public API, which returns a deduplicated *set* of tickers per text.

Concretely, both gold and predictions are reduced to a set of `(normalized_surface, label)` keys **per document** before TP/FP/FN are tallied (`benchmark.normalize_entity` + `prepare_eval_inputs`/`evaluate_model`). Normalization folds case, surrounding whitespace, and a leading `$`, so `$GME`, `GME`, `gme`, and a repeated `GME` 16× in one post all collapse to a single `("GME", "ticker")` key.

Consequences to keep in mind:
- **Do not** "fix recall" by adding more training mentions of an already-known ticker (e.g. GME/AMC). Repeated mentions are free under this metric; a false negative now means the model missed an entity in *every* context it appeared in.
- Because surfaces are normalized, boundary mismatches (`$AAPL` vs `AAPL`) no longer count as errors. `error_analysis.py` therefore reports only three categories: pure FP, pure FN, and label confusion (same normalized surface, conflicting label).
- `error_analysis.py` and `benchmark.py` share this normalization, so `errors_v{N}.json` summary numbers match the benchmark table. Error records keep the *original* surface form (not the normalized key) so `patch_test_labels.py` can still locate spans.

## Code Style

Black (line-length 88), Ruff, NumPy-style docstrings. Enforced via CI.

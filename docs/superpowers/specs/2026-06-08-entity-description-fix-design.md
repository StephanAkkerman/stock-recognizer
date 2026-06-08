# Design: Entity Description Fix + Inference-Time Relabeling

**Date:** 2026-06-08  
**Goal:** Improve combined F1 from 71% toward 80% by fixing a systematic ticker→company label confusion and sharpening training descriptions.

## Problem

Error analysis on the latest test set (bec7f714316b47c6) against the best adapter (v13) shows:

- TP=133, FP=18, FN=101 — recall (56.8%) is the bottleneck, not precision (88%)
- 93 pure false negatives vs 10 pure false positives
- 8 label confusions, **all** in the same direction: ticker predicted as company
  - Affected entities: FUBO, AMTD, SOFI, GME, NEO, COST, GOOGL, META
  - These are all-caps symbols used in trading/earnings contexts (e.g. "AMZN, GOOGL, META are ALL reporting earnings")

The current `company` description already says "MUST NOT be an uppercase ticker symbol" but the model ignores it. The descriptions don't give a reliable syntactic rule — the model falls back on semantic context (these tokens *are* companies) and guesses `company`.

## Scope

Approach C from the improvement brainstorm: description tightening + inference post-processing. No new labeled data required. Expected F1 gain: +2–4% from fixing the 8 confusions; foundation for future data additions to land cleanly.

Out of scope: new data acquisition (Approach A/B), hyperparameter search, augmentation changes.

## Design

### 1. `train.py` — `ENTITY_DESCRIPTIONS`

Replace with descriptions that give the model an unambiguous syntactic rule:

- **ticker**: an all-uppercase abbreviation (1–6 letters, with or without `$`) referring to a tradeable security. Explicitly covers the case where the abbreviation also names the company (GME, META, GOOGL).
- **company**: a name written in mixed or natural case (Apple, GameStop, Rocket Lab). Explicitly excludes all-uppercase abbreviations.

This rule is injected into every training sample via `task_to_samples()` and is seen by the model at train time alongside the span annotations.

### 2. `auto_label.py` — `SYSTEM_INSTRUCTIONS` + `FEW_SHOT_EXAMPLES`

Mirror the same syntactic rule in `SYSTEM_INSTRUCTIONS` so future auto-labeled batches carry consistent labels.

Add two new few-shot examples:
- `"GME is pumping again"` → GME=ticker (all-caps without $, but still a ticker)
- `"upstart has a decade lead in AI credit"` → upstart=company (lowercase company name)

The existing `"Tesla" → company / "TSLA" → ticker` note stays but gets extended to cover the all-caps-used-as-company pattern explicitly.

### 3. `engine.py` — post-processing in `recognize_ai()`

After the model extracts entities and before the company→ticker resolution step, apply one guard:

> If an entity is labeled `company`, its text matches `^[A-Z][A-Z0-9]{0,5}$` (all-caps, 1–6 chars), and it exists in `valid_tickers`, relabel it as `ticker`.

This is a 3–4 line addition. It does not affect `recognize()` (regex-only path).

**Edge case — AMD:** AMD is in `valid_tickers` and is sometimes gold-labeled `company`. This relabeling changes its label to `ticker`, but since `recognize_ai()` returns ticker symbols in both cases (company names get resolved to tickers via the market-data mapping), the final output is identical. The relabeling makes the internal label consistent, which matters for any downstream logic that inspects entity types.

This fix applies to production output immediately, independent of which adapter version is loaded.

### 4. Retrain

Run `python trainer/train.py` after the description changes. This produces a new adapter (v14) trained with the corrected descriptions. The post-train benchmark runs automatically and writes to `benchmark_results.json`.

### 5. Follow-up: Engine-aware benchmark mode

The current `benchmark.py` evaluates the raw model output (chunks → model → metrics). It does not apply the regex pre-pass or the post-processing in `engine.py`, so it measures model F1, not production F1.

Add an optional `--engine` flag to `benchmark.py` that routes evaluation through `StockRecognizer.recognize_ai()` instead of calling the model directly. This gives a production F1 number that reflects:
- Regex cashtag pre-pass (adds high-confidence recalls before the model runs)
- Inference-time relabeling (fixes all-caps company→ticker)
- Company→ticker resolution via market-data mapping

This is a separate task after the core changes are validated.

## Expected Impact

| Change | Errors fixed | F1 delta |
|--------|-------------|----------|
| Description fix (retrain) | 8 label confusions → TPs | +2–3% |
| Inference relabeling | Same 8 confusions in production | production only |
| Engine benchmark mode | Reveals true production F1 | measurement only |

Reaching 80% F1 will still require additional labeled data for the top missed entities (GME×9, CHTR×7, Upstart×7 — Approach A/B). These description fixes are the foundation that ensures new data lands with correct labels.

## Files Changed

| File | Change |
|------|--------|
| `trainer/train.py` | Replace `ENTITY_DESCRIPTIONS` |
| `utils/auto_label.py` | Update `SYSTEM_INSTRUCTIONS`, extend `FEW_SHOT_EXAMPLES` |
| `stock_recognizer/engine.py` | Add post-processing guard in `recognize_ai()` |
| `trainer/benchmark.py` | Add `--engine` flag (follow-up) |

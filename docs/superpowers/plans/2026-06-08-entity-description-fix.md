# Entity Description Fix + Inference Relabeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix a systematic ticker→company label confusion (8 of 101 FNs) by tightening entity descriptions in training, auto-labeling, and the engine, then retraining to measure the F1 gain.

**Architecture:** Three description strings must stay in sync — `ENTITY_DESCRIPTIONS` (train.py), `DEFAULT_LABELS` (benchmark.py), and `self.ai_labels` (engine.py) — plus the LLM prompt in auto_label.py. The engine currently passes bare label names at inference instead of descriptions; fixing that is a free improvement. Retrain runs automatically after the description changes. The `--engine` benchmark flag is a follow-up that measures production F1 (regex + model + post-processing) rather than raw model span F1.

**Tech Stack:** Python, pytest, GLiNER2 LoRA adapter, Rich (benchmark display)

---

## Files Changed

| File | Change |
|------|--------|
| `trainer/train.py` | Replace `ENTITY_DESCRIPTIONS` |
| `trainer/benchmark.py` | Replace `DEFAULT_LABELS`; add `--engine` flag + `engine_evaluate_model()` |
| `utils/auto_label.py` | Update `SYSTEM_INSTRUCTIONS`; extend `FEW_SHOT_EXAMPLES` |
| `stock_recognizer/engine.py` | Update `self.ai_labels`; fix `recognize_ai()` to pass descriptions; add relabeling guard |
| `tests/test_engine.py` | Add test verifying descriptions are passed at inference |
| `tests/test_auto_label.py` | Add tests verifying new few-shot examples appear in prompts |

---

### Task 1: Update `ENTITY_DESCRIPTIONS` in train.py

**Files:**
- Modify: `trainer/train.py:23-26`

- [ ] **Step 1: Replace `ENTITY_DESCRIPTIONS`**

  Open `trainer/train.py` and replace lines 23–26 with:

  ```python
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
  ```

- [ ] **Step 2: Run existing tests**

  ```
  pytest tests/ -v
  ```

  Expected: all previously passing tests still pass — no test covers this constant directly.

- [ ] **Step 3: Commit**

  ```bash
  git add trainer/train.py
  git commit -m "fix: sharpen ENTITY_DESCRIPTIONS — all-caps abbreviations are always tickers"
  ```

---

### Task 2: Sync `DEFAULT_LABELS` in benchmark.py

**Files:**
- Modify: `trainer/benchmark.py:41-44`

`DEFAULT_LABELS` is passed as label descriptions to `batch_extract_entities()` at evaluation time. It must match `ENTITY_DESCRIPTIONS` exactly so the model sees the same prompt it was trained on.

- [ ] **Step 1: Replace `DEFAULT_LABELS`**

  Open `trainer/benchmark.py` and replace lines 41–44 with:

  ```python
  DEFAULT_LABELS = {
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
  ```

- [ ] **Step 2: Run existing tests**

  ```
  pytest tests/ -v
  ```

  Expected: all passing.

- [ ] **Step 3: Commit**

  ```bash
  git add trainer/benchmark.py
  git commit -m "fix: sync DEFAULT_LABELS with updated ENTITY_DESCRIPTIONS"
  ```

---

### Task 3: Update `SYSTEM_INSTRUCTIONS` and `FEW_SHOT_EXAMPLES` in auto_label.py

**Files:**
- Modify: `utils/auto_label.py:123-158` (SYSTEM_INSTRUCTIONS)
- Modify: `utils/auto_label.py:64-105` (FEW_SHOT_EXAMPLES)
- Test: `tests/test_auto_label.py`

- [ ] **Step 1: Write two failing tests**

  Add to `tests/test_auto_label.py`:

  ```python
  def test_prompt_contains_allcaps_ticker_example():
      """New GME few-shot example (all-caps ticker without $) appears in the prompt."""
      prompt = auto_label.build_prompt("test input")
      assert "GME" in prompt


  def test_prompt_contains_lowercase_company_example():
      """New upstart few-shot example (lowercase company name) appears in the prompt."""
      prompt = auto_label.build_prompt("test input")
      assert "upstart" in prompt
  ```

- [ ] **Step 2: Run to confirm they fail**

  ```
  pytest tests/test_auto_label.py::test_prompt_contains_allcaps_ticker_example tests/test_auto_label.py::test_prompt_contains_lowercase_company_example -v
  ```

  Expected: FAIL — "GME" and "upstart" do not appear in the current `FEW_SHOT_EXAMPLES`.

- [ ] **Step 3: Replace the DEFINITIONS block in `SYSTEM_INSTRUCTIONS`**

  In `utils/auto_label.py`, find the `DEFINITIONS:` section (lines ~128–136) inside `SYSTEM_INSTRUCTIONS` and replace the two bullet points with:

  ```
      DEFINITIONS:
      - ticker: An all-uppercase abbreviation (1–6 letters, with or without $) for a
        tradeable security. INCLUDE the $ sign in the entity span when present.
        Label as ticker even when the abbreviation also names the company — e.g.
        GME, COST, META, GOOGL are tickers, not companies, even in earnings context.
        ETFs (SPY, QQQ, VOO, IWM), crypto (BTC, ETH), and index abbreviations used
        as tradeable (SPX, NDX) are tickers when used in a trading context.
      - company: The full or informal name of a company written in mixed or natural
        case. Can be multiple words ("Bed Bath & Beyond", "Rocket Lab", "Cash app").
        Lowercase names are still companies: "upstart", "reddit", "robinhood".
        MUST NOT be an all-uppercase abbreviation — those are tickers.
  ```

- [ ] **Step 4: Update the EDGE CASES section in `SYSTEM_INSTRUCTIONS`**

  Find the line `- A token used differently in one post can be both:` and extend the block with two new bullets after it:

  ```
      - All-caps abbreviations without $ are tickers: "GME is pumping" → GME=ticker.
        Mixed-case is company: "GameStop rallied" → GameStop=company.
      - Lowercase company names are companies: "upstart grew 40% yoy" → upstart=company.
  ```

- [ ] **Step 5: Add two entries to `FEW_SHOT_EXAMPLES`**

  Append before the closing `]` of the `FEW_SHOT_EXAMPLES` list:

  ```python
      {
          "input": "GME is pumping again, grab calls before the squeeze",
          "output": {"entities": [{"text": "GME", "label": "ticker"}]},
      },
      {
          "input": "upstart has a decade lead in AI credit scoring over traditional lenders",
          "output": {"entities": [{"text": "upstart", "label": "company"}]},
      },
  ```

- [ ] **Step 6: Run all auto_label tests**

  ```
  pytest tests/test_auto_label.py -v
  ```

  Expected: all pass including the two new tests.

- [ ] **Step 7: Commit**

  ```bash
  git add utils/auto_label.py tests/test_auto_label.py
  git commit -m "fix: update auto_label descriptions and add GME/upstart few-shot examples"
  ```

---

### Task 4: Fix engine.py — descriptions at inference + relabeling guard

**Files:**
- Modify: `stock_recognizer/engine.py:86-89` (`self.ai_labels` in `__init__`)
- Modify: `stock_recognizer/engine.py:160` (`recognize_ai` call)
- Modify: `stock_recognizer/engine.py:165-168` (add relabeling guard)
- Test: `tests/test_engine.py`

`recognize_ai()` currently calls `self.extractor.extract_entities(text, ["company", "ticker"])` — bare label names. `self.ai_labels` holds descriptions but is never used. The model was trained on descriptions, so inference should also receive them. The relabeling guard additionally reroutes all-caps valid tickers that the model labeled `company` into the ticker path, which is needed for the `--engine` benchmark mode added in Task 5.

- [ ] **Step 1: Write a failing test**

  Add to `tests/test_engine.py`:

  ```python
  def test_recognize_ai_passes_label_descriptions(recognizer, monkeypatch):
      """recognize_ai() must pass label description strings, not bare label names."""
      calls = []
      original = recognizer.extractor.extract_entities

      def capture(*args, **kwargs):
          calls.append(args[1] if len(args) > 1 else kwargs.get("labels"))
          return original(*args, **kwargs)

      monkeypatch.setattr(recognizer.extractor, "extract_entities", capture)
      recognizer.recognize_ai("$AAPL is up today")
      assert calls, "extract_entities was not called"
      assert isinstance(calls[0], dict), (
          f"expected a dict of label descriptions, got {type(calls[0])}: {calls[0]}"
      )
  ```

- [ ] **Step 2: Run to confirm the test fails**

  ```
  pytest tests/test_engine.py::test_recognize_ai_passes_label_descriptions -v
  ```

  Expected: FAIL — `calls[0]` is `["company", "ticker"]`, not a dict.

- [ ] **Step 3: Update `self.ai_labels` in `__init__` (lines 86–89)**

  Replace the four stale lines with:

  ```python
              self.ai_labels = {
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
  ```

- [ ] **Step 4: Fix the `extract_entities` call in `recognize_ai()` (line 160)**

  Before:
  ```python
          result = self.extractor.extract_entities(text, ["company", "ticker"])
  ```

  After:
  ```python
          result = self.extractor.extract_entities(text, self.ai_labels)
  ```

- [ ] **Step 5: Add the relabeling guard after `entities` is assigned**

  Find the block starting with `entities = result.get(...)` and the line `all_ai_mentions = entities.get("company", []) + entities.get("ticker", [])`. Replace those two lines with:

  ```python
          entities = result.get("entities", result) if isinstance(result, dict) else {}

          # Promote all-caps valid-ticker entities mislabeled as company → ticker.
          # The model occasionally labels symbols like GME/META as company in earnings
          # contexts. The resolved output is identical either way (valid_tickers check
          # catches them downstream), but correct labeling is needed for --engine mode.
          _all_caps_re = re.compile(r"^[A-Z][A-Z0-9]{0,5}$")
          promoted = [
              m for m in entities.get("company", [])
              if _all_caps_re.match(str(m)) and str(m) in self.valid_tickers
          ]
          company_entities = [m for m in entities.get("company", []) if m not in promoted]
          ticker_entities = list(entities.get("ticker", [])) + promoted

          # Flatten all AI entities into one list to resolve
          all_ai_mentions = company_entities + ticker_entities
  ```

- [ ] **Step 6: Run all tests**

  ```
  pytest tests/ -v
  ```

  Expected: all pass including `test_recognize_ai_passes_label_descriptions`.

- [ ] **Step 7: Commit**

  ```bash
  git add stock_recognizer/engine.py tests/test_engine.py
  git commit -m "fix: pass ai_labels descriptions to extractor and add company→ticker relabeling guard"
  ```

---

### Task 5: Retrain and verify improvement

- [ ] **Step 1: Run training**

  ```bash
  python trainer/train.py
  ```

  Expected: prints train/val sample counts, runs training with early stopping (typically stops at epoch 7–9 of 10), then automatically benchmarks on the test set and writes to `models/benchmark_results.json`. Wait ~30–60 min on GPU.

- [ ] **Step 2: Check the new adapter's F1**

  The final line of train.py's output shows overall P/R/F1 for the new adapter vs the test set. Compare to the current best (v4: F1=71.5% on test set `bec7f714316b47c6`).

  If the new adapter's F1 is lower than v4, run the full benchmark to compare all versions:

  ```bash
  python trainer/benchmark.py
  ```

- [ ] **Step 3: Commit the training metadata**

  The adapter weights are git-ignored. Check if `training_metadata.json` is tracked:

  ```bash
  git status models/
  ```

  If `models/reddit_adapter_v14/training_metadata.json` is untracked, add it:

  ```bash
  git add models/reddit_adapter_v14/training_metadata.json
  git commit -m "train: v14 with tightened entity descriptions (ticker/company distinction)"
  ```

---

### Task 6: Add `--engine` flag to benchmark.py

This adds a second evaluation path that routes inference through `StockRecognizer.recognize_ai()`, measuring production F1 (regex cashtag pre-pass + model + relabeling guard + company name resolution) rather than raw span-level NER F1.

**Files:**
- Modify: `trainer/benchmark.py`

- [ ] **Step 1: Add `engine_evaluate_model()` after `evaluate_model()`**

  Insert the following function after the `evaluate_model()` function (before `prepare_eval_inputs`):

  ```python
  def engine_evaluate_model(adapter_path, dataset, model_name="Engine"):
      """Evaluate production F1 via StockRecognizer.recognize_ai().

      Unlike evaluate_model(), this path includes the regex cashtag pre-pass,
      the company→ticker relabeling guard, and company name resolution, giving
      the real-world ticker-level P/R/F1 rather than span-level NER metrics.

      Gold annotations (both ticker and company spans) are resolved to ticker
      symbols using the same engine logic so both sides are comparable.
      """
      import sys as _sys
      _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
      from stock_recognizer.engine import StockRecognizer

      recognizer = StockRecognizer(use_ai=True, adapter_path=adapter_path)

      tp = fp = fn = 0
      for entry in dataset:
          predicted = set(recognizer.recognize_ai(entry["text"]))

          # Resolve gold spans to ticker symbols using the same engine logic so
          # both sides of the comparison speak the same language.
          gold_tickers = set()
          for e in entry["entities"]:
              span_text = entry["text"][e["start"]:e["end"]]
              cleaned = recognizer._clean_token(span_text)
              if cleaned in recognizer.valid_tickers:
                  gold_tickers.add(cleaned)
              else:
                  base = cleaned.split()[0]
                  resolved = recognizer.company_to_ticker.get(
                      cleaned, recognizer.company_to_ticker.get(base)
                  )
                  if resolved:
                      gold_tickers.add(resolved)

          tp += len(predicted & gold_tickers)
          fp += len(predicted - gold_tickers)
          fn += len(gold_tickers - predicted)

      return {"overall": calculate_metrics(tp, fp, fn)}
  ```

- [ ] **Step 2: Add argument parsing and `--engine` branch to `__main__`**

  At the top of the `if __name__ == "__main__":` block, insert:

  ```python
      import argparse as _argparse
      _parser = _argparse.ArgumentParser()
      _parser.add_argument(
          "--engine", action="store_true",
          help="Also evaluate via StockRecognizer.recognize_ai() for production F1."
      )
      _args = _parser.parse_args()
  ```

  Then after the existing table is printed (after `console.print(_render_params(rows))`), add:

  ```python
      if _args.engine:
          console.print("\n[bold cyan]Engine-mode evaluation (production F1)[/bold cyan]")
          console.print("[dim]Includes regex pre-pass, relabeling guard, company resolution[/dim]")
          for name, adapter_path in model_configs:
              if adapter_path is None:
                  continue
              engine_scores = engine_evaluate_model(adapter_path, dataset, model_name=name)
              overall = engine_scores["overall"]
              console.print(
                  f"  {name}: "
                  f"P={overall['p']:.2%}  R={overall['r']:.2%}  F1={overall['f1']:.2%}"
              )
  ```

- [ ] **Step 3: Run to verify**

  ```bash
  python trainer/benchmark.py --engine
  ```

  Expected: normal benchmark table prints first, then the engine-mode section appears for each adapter with a production F1 that includes the regex and resolution effects.

- [ ] **Step 4: Commit**

  ```bash
  git add trainer/benchmark.py
  git commit -m "feat: add --engine flag to benchmark for production F1 via StockRecognizer"
  ```

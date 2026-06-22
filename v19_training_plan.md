# v18 → v19 Training Plan

Source: `errors_v18.json` (adapter v18, threshold 0.75, test folder `data/test`),
re-scored under the **set-based, deduplicated-per-document** metric
(`benchmark.normalize_entity` — an entity counts as caught if found at least once
in a post). See the "Scoring contract" section in `CLAUDE.md`.

> Note: `next_training_plan.md` still holds the **v17** analysis and is stale —
> this file supersedes it for the v19 cycle.

## Headline — the metric fix changed the story

Re-scoring v18 the way the engine actually behaves (a post that says GME 16× needs
GME found *once*) flips the diagnosis:

| metric | old per-occurrence | set-based | **+ P2 gold fixes** |
|---|---|---|---|
| precision | 0.826 | 0.801 | **0.804** |
| recall | 0.553 | 0.906 | **0.925** |
| F1 | 0.662 | 0.850 | **0.860** |
| TP / FP / FN | 365 / 77 / 295 | 270 / 67 / 28 | **271 / 66 / 22** |

**Recall is not the problem — it's 92%.** The model already finds almost
everything Reddit talks about. The 295 "false negatives" were overwhelmingly
repeated mentions of tickers (GME, AMC, $FUTU) the model caught at least once;
they vanished under dedup (FN: 295 → 28), and the P2 gold fixes took it to 22.

**Precision looks like the weaker side at 80% — but read the next section first.**
The earlier P0 ("add training data for GME/AMC/Velo3D/etc.") is **retired**: those
were measurement artifacts, and adding more mentions of known tickers buys nothing
under the dedup metric.

## ⚠️ Raw-model FP ≠ production FP — measure the engine

`error_analysis.py` scores the **raw GLiNER model**. But the product is
`recognize_ai()`, which drops `AMBIGUOUS_WORDS` and keeps only mentions that
**resolve to a ticker**. Replicating that filter on the 63 raw FPs:

- **~46 are dropped by the engine** (don't resolve to anything) — e.g. `Burry`,
  `Monkeypox`, `EPYC`, `DRAM`, `Murica`, `Melvin`, most of doc 62's jargon. They
  inflate raw-model FP but **never reach production output**.
- **17 resolve to a ticker; 2 of those (`TSLA`→TSLA, `QXO`→QXO) are correct**
  (already in gold). So **≈15 are true production FPs**, and they fall into two
  fixable patterns:

  **(1) Common word → obscure ticker** — `financial`→FISI, `capital`→CBNK,
  `stock`→SYBT, `strategic`→STRA, `Tenet`→THC, `fico`→FICO, `SPX`→FLOW,
  `RHs`→RHS. None are in `AMBIGUOUS_WORDS` (but `here`/`now`/`dyor` already are).

  **(2) Multi-word jargon → first-word collision** — `Laser Powder Bed
  Fusion`→LASE, `Rapid Production Solutions`→RPID, `Andretti Global`→POLE,
  `Big Pharma Links`→BCYP, `United Waste`→PRKS. The engine resolves a phrase by
  `base_name = m_clean.split()[0]`. Notably these phrases ARE in
  `patch_test_labels.COMPANY_BLOCKLIST` — but **that blocklist is not used by the
  engine.**

**UPDATE — measured, and then fixed.** The engine (`--engine`) benchmark showed
production precision was **41%** (stuck there across all 18 versions), confirming
this is an *engine* problem, not a model one. The P0 fix below took it to
**87.8%** (F1 58%→93%). Raw-model precision (0.80) is a training diagnostic, not
the product number.

## Error composition (raw model)

| category | count |
|---|---|
| pure FP (raw) | 63 → **≈15 reach production** |
| pure FN (genuinely missed everywhere) | 19 |
| label confusion | 3 |

The precision work is now mostly an **engine-resolution** problem, not a
retraining one.

## Doc 62 (Velo3D) — big in raw FP, mostly harmless in production

| doc | errors | FP | FN | topic |
|---|---|---|---|---|
| **62** | **18** | **16** | 2 | Velo3D ($VELO) DD |
| 41 | 5 | 4 | 1 | "The AI Market: A Gradual Rotation" |
| 49 | 4 | 2 | 2 | $PATH |
| 67 | 4 | 1 | 3 | RKLB |
| 70 | 4 | 4 | 0 | RobinHood class-action |

Doc 62 holds 16 of 66 raw FPs, but per the section above **most don't reach
production** (`Murica`, `defense & aerospace`, `IDIQ`, `CA HQ`, `EPYC`, `LPBF`,
`nufacturers`, `significant capital`… resolve to nothing). The handful that *do*
leak (`Laser Powder Bed Fusion`→LASE, `Rapid Production Solutions`→RPID,
`Andretti Global`→POLE) are first-word resolution collisions — fixed at the
engine, not by re-annotating the doc. So doc 62 is **raw-model hygiene, not the
precision emergency it looked like.** Hard-negative mining it is now P3, not P0.

## False positives, bucketed (the precision work)

**A. Genuine hallucinations — but harmless unless they resolve.** Words/slang/
acronyms the raw model emits. Only the ones that resolve to a ticker (see ⚠️
section) actually hurt; the rest are cosmetic raw-model noise:

```
here ×3, financial ×2, spreadbet, Fed, chest, fundamentals, capital,
Southeast Asia, starlink (SpaceX product), Monkeypox, CHYNA (slang),
CSRC (China regulator), EOY, DYOR, $ AUG (option month)
```

**B. Looked like under-labeling, but ISN'T (verified against the data, P2).**
On inspection these are *not* free-precision promotions:

- `TSLA` — gold already has `TSLA/ticker` @152. The model emitted TSLA twice
  (ticker = TP, *and* company = the FP). A model dup-label error, not a gold gap.
- `SPX` — S&P 500 index; policy says skip indices, not in `valid_tickers` →
  genuine hallucination → **move to Bucket A (hard negative)**.
- `MPVX` — not in `valid_tickers`, can't resolve → genuine FP → **Bucket A**.
- `Quantum Emotion` — this one *was* real, but it's a malformed-gold-span bug,
  handled in C (fixed in P2), not a promotion.

`patch_test_labels.py` is **not usable** here — it skips all of the above and
surfaces only junk (`here`, `capital`, `financial`, `stock`, `Now`, `DYOR`…)
that coincidentally matches obscure DB symbols. Do not `--auto` it.

**C. Malformed / incorrect gold spans — fixed by hand (P2, DONE).**

- ✅ `Quantum Emotion ($QNC` gold span had mismatched offsets (`text` said
  "Quantum Emotion Corp." but `518:539` covered "Quantum Emotion ($QNC").
  Re-spanned to `518:533` = `Quantum Emotion`; the `$QNC/ticker` @535 already
  existed. Model's clean `Quantum Emotion` now matches → −1 FP, −1 FN.
- `Azure OpenAI` (pred) vs `OpenAI` (gold) — model grabbed the qualifier; gold
  `OpenAI` is correct. This is a **model boundary error, not a gold bug** → P1.

## False negatives (now just 19, post-P2 — small and clean)

**Test-set labeling errors — fixed in P2 (DONE):**
- ✅ `stock`/company removed; ✅ `Quantum Emotion` re-spanned; ✅ 4 SpaceX spans
  de-punctuated.

**Genuine vocabulary gaps (real, but few):**
```
TYDE, BBBY, Lunr, 6702.T (Japanese ticker), AGIX (Anthropic-ETF),
Dymatize, Premier Protein, Yomiuri Shimbun, UiPath Labs, Salesforce,
Ray-Ban, Reddit (RDDT), DOW (index)
```

**Form gaps / notable misses:**
- `$FLY` — a **cashtag** missed ("[$FLY] me to the moooon"); the wordplay context
  ("fly me to the moon") likely masked it. Cashtags should be near-100% — worth a
  look.
- `meta` lowercase ("as [meta] has been spending") — case generalization.
- `Velo` / `Snap` — short-forms; also appear in the confusion table.
- `GME` used as a verb ("[GME] that money!") — arguably shouldn't be gold.

## Label confusions (3) — NOT a data-consistency problem (verified, P2)

```
SOFI  gold ticker → pred company
EOS   gold ticker → pred company
Snap  gold company → pred ticker
```

Checked: training labels are **already policy-consistent** (`SOFI`=ticker ×15,
`SoFi`=company ×4, `SNAP`=ticker ×5, `Snap`=company ×4, `snap`=ticker ×3), and
the test gold is correct per policy. So `fix_label_policy.py` finds nothing —
these are plain model errors on hard instances, not bad labels.

The one actionable root cause: **`EOS` has zero training examples**, so the model
has no signal and defaults to company. That's a P3 training gap, not a fix here.
(`SOFI`/`Snap` errors despite ample, consistent data → just need more/varied
contexts.)

---

## Concrete actions for v19, in priority order

Versions auto-increment: `train.py::get_next_version()` creates
`models/reddit_adapter_v19/` and self-benchmarks. No manual bump.

### P0 — Engine precision fix ✅ MOSTLY DONE — production F1 58% → 93%

The real problem was the **engine**, not the model: precision sat at ~41% across
*all 18 versions* while the regex path uppercased the whole post and matched every
prose word (`don't`→DON, `edit`→EDIT, `away`→AWAY) against `valid_tickers`.

- [x] **Case-aware matching** (`engine.py`): `ticker_re` now matches an uppercase
      2–6 letter core (+ optional plural/possessive `s`) against the **original**
      text, not an uppercased copy. Lowercase prose no longer matches; `AAPLs`/
      `MSFT's` still do. Regex-only precision **34% → 83%** (FP 214 → 19).
- [x] **`AMBIGUOUS_WORDS` additions** (`constants.py`): caps acronyms (`MIN`,
      `EDIT`, `GPU(S)`, `EPS`, `PDT`, `RSI`, `RSU`, `WTF`, `EUV`, `HBM`, `IEEPA`,
      `DLA`, `FX`, `HQ`) + AI-path word collisions (`FINANCIAL`, `STOCK`,
      `STRATEGIC`) + `SPX` (index). This also fixes the AI path, which checks
      `AMBIGUOUS_WORDS`.
- [x] **Verified**: engine (production) v18 **P=41.4%→87.8%, R=98.5%→98.0%,
      F1=58.3%→92.6%**. `test_financial_jargon_shield` now passes; no regressions.
- [ ] **Remaining: multi-word → first-word over-resolution** (the residual lever).
      `recognize_ai` resolves a phrase by `base_name = m_clean.split()[0]`, so
      `Laser Powder Bed Fusion`→LASE, `United Waste`→PRKS, `American Express`→AAL.
      This still powers `test_very_long_post` extras (AAL, PRKS, CAPE) and the
      doc-62 leaks. Tightening it is delicate — base_name also gives recall
      (`Micron`→MU) — so require a fuller match or port the
      `patch_test_labels.COMPANY_BLOCKLIST` phrases into the engine. Needs care.
- [ ] Pre-existing, unrelated: `test_ethe_post` fails because **`ETHE` isn't in
      `valid_tickers`** (market-data gap), and `test_reddit_post` leaks `EVEX`
      (AI hallucination of "eve"). Both are separate from the regex fix.

### P1 — Hard-negative mining (only for FPs that actually resolve)
- [ ] Target the ~15 production-leaking FPs, not all 63 raw FPs. Add negative/
      unlabeled contexts so the *model* stops emitting `financial`, `capital`,
      `stock`, `strategic`, `Tenet`, `SPX`, plus the doc-62 phrases that resolve
      (`Laser Powder Bed Fusion`, `Rapid Production Solutions`, `Andretti Global`).
- [ ] Skip hard-negatives for the ~46 non-resolving raw FPs (`Burry`, `Monkeypox`,
      `Murica`, `EPYC`, `DRAM`, …) unless they're cheap to include — they don't
      hurt production.
- [ ] Reuse the `negatives_mined.json` workflow (`data/labeled/negatives_mined.json.bak`).

### P2 — Test-set hygiene ✅ DONE
- [x] Removed bogus `stock`/company annotation (task 514, labeled + test).
- [x] Re-spanned malformed `Quantum Emotion ($QNC` → `Quantum Emotion` (task
      8000187, labeled + test); `$QNC/ticker` already present.
- [x] Trimmed trailing punctuation on 4 SpaceX spans (`SpaceX.`, `SpaceX).`,
      `SpaceX?`, `Spacex.`; tasks 738/751, labeled + test) → model's clean
      `SpaceX` now matches.
- [x] Repo-wide malformed-span scan: data is otherwise clean. Stragglers left for
      review: `SPC(E)` (ambiguous), `Duke Energy.` (clear), `Warner Bros.` /
      `metro inc.` (legit abbreviation periods, leave).
- [x] Verified the rest are NOT test-hygiene: `patch_test_labels` surfaces only
      junk here; `TSLA`/`SPX`/`MPVX` are not clean promotions (see Bucket B); the
      3 confusions are model errors on policy-correct gold (see above).
- [x] Raised `error_analysis.py --top` default 20→50 and added `--top 0` =
      unlimited, so all FN/FP rows show in future runs.
- Net effect: ≈ −1 FP, −4 FN. Augmented copies of edited tasks still hold the old
  versions — regenerate in P4 (do not hand-edit swapped text).

### P3 — Raw-model hygiene + recall/confusion top-ups (low priority)
- [ ] Doc 62 (Velo3D) hard-negative pass — *demoted from P0*; cleans the raw model
      but most of its FPs don't reach production.
- [ ] **Add `EOS` training examples** (currently zero) — the only confusion with a
      real data root cause.
- [ ] Check the `$FLY` cashtag miss — cashtag recall should be near-100%.
- [ ] Add a few examples for the genuine vocab gaps: `Salesforce`, `UiPath Labs`,
      `Lunr`, `Ray-Ban`, `Reddit`/RDDT, `6702.T`, `AGIX`. Skip index/verb edge
      cases (`DOW`, verb-`GME`).

### P4 — Rebuild + train + verify
- [ ] `python utils/augment_data.py` to regenerate `data/augmented/`.
- [ ] `python trainer/validate_descriptions.py` (ticker vs company descriptions).
- [ ] `pytest` + `ruff check .` clean.
- [ ] `python trainer/train.py` → v19, auto-benchmark.
- [ ] `python trainer/error_analysis.py --adapter v19 --save-json errors_v19.json`
      (raw-model diagnostic) **and** `python trainer/benchmark.py` (engine/product
      metric). **Primary target: production (engine) precision** via the P0 fixes —
      most of which need no retrain. Hold recall ≥ 0.92 (raw); F1 goal > 0.87.

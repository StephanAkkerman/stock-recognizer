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

| metric | old per-occurrence | **set-based (true)** |
|---|---|---|
| precision | 0.826 | **0.801** |
| recall | 0.553 | **0.906** |
| F1 | 0.662 | **0.850** |
| TP / FP / FN | 365 / 77 / 295 | **270 / 67 / 28** |

**Recall is not the problem — it's 91%.** The model already finds almost
everything Reddit talks about. The 295 "false negatives" were overwhelmingly
repeated mentions of tickers (GME, AMC, $FUTU) the model caught at least once;
they vanished under dedup (FN: 295 → 28).

**Precision (80%) is now the weaker side and the v19 target.** The earlier P0 —
"add training data for GME/AMC/Velo3D/etc." — is **retired**: those were
measurement artifacts, and adding more mentions of known tickers buys nothing
under the dedup metric.

## Error composition

| category | count |
|---|---|
| pure FP (hallucination) | 64 |
| pure FN (genuinely missed everywhere) | 25 |
| label confusion | 3 |

FP now outnumbers FN ~2.4:1. The work is precision.

## The single biggest lever: doc 62 (Velo3D)

| doc | errors | FP | FN | topic |
|---|---|---|---|---|
| **62** | **18** | **16** | 2 | Velo3D ($VELO) DD |
| 41 | 5 | 4 | 1 | "The AI Market: A Gradual Rotation" |
| 49 | 4 | 2 | 2 | $PATH |
| 67 | 4 | 1 | 3 | RKLB |
| 70 | 4 | 4 | 0 | RobinHood class-action |

One document holds **16 of 67 total FPs (24%)**. Every other doc is now ≤4 errors.
Doc 62 is over-predicting company/ticker entities en masse (likely Velo3D product
names, internal divisions, and manufacturing jargon tagged as companies). Fixing
this one doc's training/test annotations is the highest-ROI action available.

## False positives, bucketed (the precision work)

**A. Genuine hallucinations — need hard negatives.** Words, slang, acronyms and
non-tradeable entities the model should suppress:

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

## False negatives, bucketed (small, mostly not training gaps)

**Test-set labeling errors — fix gold, don't train (P2, DONE):**
- ✅ `stock` labeled `company` ("own the [stock]", task 514) — not an entity, removed.
- ✅ `Quantum Emotion ($QNC` malformed span (task 8000187) — re-spanned (see C).

**Genuine vocabulary gaps (real, but few):**
```
TYDE, BBBY, Lunr, $FLY, 6702.T (Japanese ticker), DOW (index),
Dymatize, Premier Protein, Yomiuri Shimbun, UiPath Labs, Salesforce,
Ray-Ban, Reddit (RDDT)
```

**Edge cases / form gaps:**
- `meta` lowercase ("as [meta] has been spending") — case generalization.
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

### P0 — Doc 62 (Velo3D): kill the 16-FP cluster
- [ ] Manually review doc 62 in both `data/test/` and `data/labeled/`. Identify
      which Velo3D product names / divisions / jargon the model tags as
      entities, and add them as **hard negatives** (unlabeled context) in
      training. This one fix addresses ~24% of all FPs.

### P1 — Precision: hard-negative mining (Bucket A)
- [ ] Add negative/unlabeled contexts for the genuine hallucinations:
      `here`, `financial`, `Fed`, `chest`, `fundamentals`, `capital`,
      `Southeast Asia`, `starlink`, `Monkeypox`, `spreadbet`, `CHYNA`,
      plus `SPX` (index) and `MPVX` (unresolvable) reclassified from Bucket B.
- [ ] Acronyms/bodies (`CSRC`, `EOY`, `DYOR`, `$ AUG`) — these already have
      blocklist entries in `patch_test_labels.py`; verify the *engine* suppresses
      them and consider extending `AMBIGUOUS_WORDS` / description text rather than
      relying on the model alone.
- [ ] `Azure OpenAI`→`OpenAI` boundary over-grab and the `TSLA/company` dup-label
      are model errors — varied training contexts, not data fixes.
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

### P3 — Targeted recall / confusion top-ups (low priority — recall is 91%)
- [ ] **Add `EOS` training examples** (currently zero) — the only confusion with a
      real data root cause.
- [ ] Add a few examples for the genuine vocab gaps: `Salesforce`, `UiPath Labs`,
      `Lunr`, `Ray-Ban`, `Reddit`/RDDT, `6702.T`. Skip index/verb edge cases
      (`DOW`, verb-`GME`).

### P4 — Rebuild + train + verify
- [ ] `python utils/augment_data.py` to regenerate `data/augmented/`.
- [ ] `python trainer/validate_descriptions.py` (ticker vs company descriptions).
- [ ] `pytest` + `ruff check .` clean.
- [ ] `python trainer/train.py` → v19, auto-benchmark.
- [ ] `python trainer/error_analysis.py --adapter v19 --save-json errors_v19.json`
      and compare against this table. **Target: lift precision from 0.80 toward
      0.85+ without dropping recall below ~0.90.** F1 goal: > 0.87.

What the errors actually show
The dominant issues are:

1. GME/AMC annotation inconsistency (massive FN source)
GME is the single most-missed entity (14 FNs) but also appears as a FP (2x as company) and in label confusion (both ticker→company and company→ticker). Same with AMC (10 FNs, 3x label confusion). The model has been trained on contradictory labels for the same tokens — sometimes ticker, sometimes company. The training signal is incoherent for these. Fix the annotations to be consistent (ticker when used as a stock identifier, company when referenced as a business entity), and these FNs will largely resolve.

2. Possessives account for 7/8 boundary mismatches
"Tesla's", "Meta's", "Robinhood's", "SpaceX's", "$GRAB's" — the model predicts the possessive form, gold has the bare form (or vice versa). Each of these counts as 1 FP + 1 FN, inflating your error count artificially. The fix is in the training preprocessor: strip trailing 's and ' from entity spans at annotation-load time in train.py:86-89.

3. Doc 63 (Velo3D) drags down recall by itself
32 errors, 19 FNs, mostly Velo3D. One heavily-cited post the model has never seen enough of. Label more Velo3D posts.

4. FP categories are clear and fixable
The 96 FPs cluster into distinct groups you can address:

Reddit slang: DYOR, NFA, AH, MMs, Apes, PSA — add to AMBIGUOUS_WORDS or put in entity description
Option expiry months: $ AUG, Aug — the model sees $ prefix and fires; add to entity description exclusions
Financial bodies/media: Bloomberg, FINRA, DTC, RSA, CSRC — need hard-negative examples with these in context
Noise/typos: amirite, gae bear, artice, fukd, Pamp — add targeted hard negatives
5. Big-name tech companies are systematically missed
NVIDIA (9), Robinhood (8), Meta (7), Microsoft (6), CoreWeave (6), Google (4), GitHub (4), Anthropic (4), OpenAI (2) — these are household names appearing in multiple posts. The model simply hasn't seen enough examples of them in financial discussion context.

Concrete action plan (prioritized by FN impact)
Step 1 — Fix annotation data (before next train run):

Audit all GME and AMC labels across data/labeled/ — decide the canonical rule and fix inconsistencies. These 2 entities alone account for ~30 FNs.
Strip possessives from entity spans during loading in task_to_samples() — a 3-line fix that converts 7 boundary mismatches into TPs.

Step 2 — Target the hotspot documents:
Run the dataset builder on posts similar to docs 63, 41, 45, 29, 54 (Velo3D, AI rotation, KYTX, SpaceX/RKLB, AMC/GME). These 5 documents contribute roughly 110 of your 238 FNs. Labeling 5-10 posts in each topic area would have outsized impact.

Step 3 — Expand AMBIGUOUS_WORDS and entity descriptions:
Add to AMBIGUOUS_WORDS in constants.py: AUG, DYOR, NFA, GMV, AH, DTC, PSA (if not already there). Update the entity descriptions to explicitly call out "MUST NOT be option expiry months, Reddit slang acronyms (DYOR, NFA, DD), or financial regulatory bodies (SEC, FINRA, DTC)."

Step 4 — Mine targeted hard negatives:
Add Bloomberg, FINRA, DTC, RSA, CSRC to SEED_FPS in mine_hard_negatives.py:53 and re-run with --n 400.

The possessive fix and GME/AMC annotation cleanup are the highest-leverage changes because they're pure measurement artifacts — fixing them will recover 30+ FNs without labeling a single new document. Want me to implement those two?
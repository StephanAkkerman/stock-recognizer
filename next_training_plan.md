v17 Error Analysis
The headline stat tells the story: precision is fine (81%) but recall is terrible (56%). You have 275 FNs vs only 82 FPs. The model is under-predicting far more than it's hallucinating.

Issues split into three buckets:
Bucket 1: Test-set ground truth is wrong (inflating FP count artificially)

The model is predicting things that are correct, but your gold labels are missing them:

pltr → PLTR is Palantir's actual ticker - this is a real hit, not a hallucination
Micron → Micron Technology (MU) is clearly a stock reference in context ("Micron blasted 20%")
Bloomberg → legitimate financial company reference (Bloomberg L.P.)
Morningstar → investment research company (MORN)
FREENOW → real company acquisition being discussed
COX → Cox Communications, relevant because it's merging with CHTR
Boundary mismatches: saveAMC→AMC, Google Gemini→Google, Azure OpenAI→OpenAI, Ray-Ban AI→Ray-Ban
These should be re-labeled as correct in your test set, which will move your real FP count down ~10-15.

Bucket 2: Noisy training data is teaching wrong behaviors (real FPs)

The model is predicting clearly wrong things - this comes from noisy annotations in training:

Might, fukd, go, air, ish, amirite as tickers
pink slip, artice, SG&A, convexity, Defined capital as companies
PDT, PT, EOY, RSUs, EV, NDAA, MEXT, METI as tickers (all abbreviations/acronyms for concepts, not stocks)
ChatGPT, Burry, reddit, The Verge as companies (products/people/sites, not listed companies)
The worst offender is doc 63 (Velo3D post): 12 FP + 18 FN = 30 errors in a single document. That doc is probably badly labeled and is actively hurting training.

Bucket 3: Recall gaps - model consistently misses core entities

These appear in FN over and over:

Entity	Missed	Why this matters
GME	15×	Most-discussed meme stock
AMC	12×	Same
Velo3D	12×	One noisy doc pulling all weight
$FUTU	10×	Dollar-prefixed variant not generalized
Robinhood	9×	Name form never learned
NVIDIA	9×	Most-discussed AI stock
Meta	8×	Label confusion: labeled company but predicted ticker (it's both)
space X	8×	Two-word non-standard spelling
Concrete actions, in priority order:
Re-label doc 63 (Velo3D post) - it's your single biggest training/test liability with 30 errors. Likely has a lot of mis-annotated spans.

Fix test set ground truth - go through the FP list and move the correct predictions (pltr, Micron, Bloomberg, Morningstar, etc.) into gold annotations. Your real precision is likely higher than 81%.

Add training examples for GME, AMC, NVIDIA, Meta, Robinhood - these are high-recall misses you can fix by scraping posts mentioning them and labeling/augmenting.

Add negative examples for financial abbreviations - documents with PDT, EV (Enterprise Value), SG&A, RSUs, PT (Price Target) that are not labeled as tickers. The model needs to see these in context without labels to stop firing on them.

Decide on your labeling policy for Meta / CSRC / FUTU - these appear in label confusion. Pick one label (ticker or company) and be consistent across all documents. The current inconsistency is confusing the model.

The biggest bang for your buck is probably #2 (fixing test set accuracy so you know your real numbers) followed by #3 (adding training data for the most-missed high-frequency stocks).
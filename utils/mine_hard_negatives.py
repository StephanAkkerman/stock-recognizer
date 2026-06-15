"""Mine hard-negative training examples from scraped Reddit data.

Hard negatives are posts that contain *ticker-shaped* tokens (`BUY`, `CNBC`,
`JPOW`, `YOLO`, ...) but no real entities. Training on them teaches the model
that capitalized 2-6 letter words aren't automatically entities — directly
attacking the FP modes surfaced by `trainer/error_analysis.py`.

Run:
    python utils/mine_hard_negatives.py                       # default: 300 negatives
    python utils/mine_hard_negatives.py --n 500
    python utils/mine_hard_negatives.py --dry-run             # print stats, write nothing
    python utils/mine_hard_negatives.py --output data/labeled/negatives_v2.json

What "hard-negative" means here:
  - The text contains at least one token that looks ticker-shaped or is a
    known false-positive seed.
  - `StockRecognizer.recognize()` (regex path) returns `[]` for the text.
  - No first-word of any known company in `company_to_ticker` appears
    as a substring.

Together these gates aim to keep mined posts genuinely entity-free while
still being maximally tempting for the model. The risk is that an obscure
ticker the engine doesn't know slips through — but those are rare and the
training signal is dominated by the explicit no-entity supervision.
"""

import argparse
import glob
import hashlib
import json
import os
import random
import re

import financedatabase as fd
import pandas as pd
from rich.console import Console
from rich.table import Table

from stock_recognizer.constants import AMBIGUOUS_WORDS, EXCHANGE_BLACKLIST
from stock_recognizer.engine import StockRecognizer

try:
    from utils.cleaner import clean_reddit_markdown
except ImportError:
    from cleaner import clean_reddit_markdown

console = Console()

# Tokens v4 specifically hallucinated on, plus high-risk Reddit slang the
# model is likely to over-fire on. Each seed match contributes more score
# than a generic ambiguous-word match.
SEED_FPS = {
    "CNBC", "ER", "EUROPOORS", "STOCK",   # from v4 error_analysis output
    "BUY", "SELL", "HOLD", "DUMP", "PUMP",
    "YOLO", "DD", "FUD", "FOMO", "PUTS", "CALLS",
    "JPOW", "POWELL", "FED", "SEC", "IRS",
    "ATH", "ATL", "NFA", "IMO", "TLDR", "LMAO", "AMA",
    "OP", "MOD", "USA", "UK", "CEO", "CFO", "IPO",
    "BULL", "BEAR", "MOON", "GAINS", "LOSS", "EARNINGS",
    # Financial bodies / media that produce FPs (v15 error analysis)
    "FINRA", "DTC", "RSA", "CSRC", "BLOOMBERG",
}

CASHTAG_RE = re.compile(r"\$[A-Za-z]{1,6}\b")
TICKER_SHAPED_RE = re.compile(r"\b[A-Z]{2,6}\b")

# Mining starts task IDs here — well above any existing labeled task ID so
# the deterministic train/val split treats these as new tasks without
# colliding with real annotations.
ID_OFFSET = 9_000_000


def normalize_for_dedup(text):
    """Hash the first 200 cleaned chars to detect duplicates against labeled data."""
    if not text:
        return None
    norm = re.sub(r"\s+", " ", text.strip().lower())[:200]
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def collect_seen_hashes(folders):
    """Build a set of dedup hashes from every Label Studio export in `folders`."""
    seen = set()
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for fp in glob.glob(os.path.join(folder, "*.json")):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            for task in data:
                text = task.get("data", {}).get("text")
                h = normalize_for_dedup(text)
                if h:
                    seen.add(h)
    return seen


def score_text(text, valid_tickers):
    """Hard-negative tempting-ness. Returns (score, matched_seeds, ticker_shaped_count)."""
    if not text:
        return 0.0, [], 0
    matched_seeds = []
    score = 0.0
    ticker_shaped_count = 0
    for tok in TICKER_SHAPED_RE.findall(text.upper()):
        ticker_shaped_count += 1
        if tok in SEED_FPS:
            score += 2.0
            matched_seeds.append(tok)
        elif tok in AMBIGUOUS_WORDS:
            score += 1.0
        elif tok not in valid_tickers:
            score += 0.3
    return score, matched_seeds, ticker_shaped_count


def has_known_ticker(text, valid_tickers):
    """Direct scan for any 2-6 letter token matching a real ticker.

    Stricter than ``StockRecognizer.recognize()``, which short-circuits on
    mostly-uppercase posts. For mining negatives we want zero leakage of real
    tickers, even from shouty WSB titles like "SELL SPY NOW".
    """
    upper = text.upper()
    for tok in TICKER_SHAPED_RE.findall(upper):
        if tok in valid_tickers and tok not in AMBIGUOUS_WORDS:
            return True
    return False


def likely_has_company(text, company_first_words):
    """True if any known company first-word appears as a standalone uppercase token."""
    upper = text.upper()
    # Use word-boundary check to avoid 'APPLE' matching inside 'APPLES' or 'APPLEBEE'
    for name in company_first_words:
        if len(name) < 4:
            continue
        if re.search(rf"\b{re.escape(name)}\b", upper):
            return True
    return False


def load_sources(paths):
    """Yield raw text strings from one or more CSVs. Auto-detects schema."""
    for path in paths:
        if not os.path.exists(path):
            console.print(f"[yellow]Skipping missing source: {path}[/yellow]")
            continue
        df = pd.read_csv(path)
        cols_lower = {c.lower(): c for c in df.columns}
        text_col = cols_lower.get("text") or cols_lower.get("body")
        title_col = cols_lower.get("title")
        if not text_col:
            console.print(f"[yellow]No 'text' column in {path}; skipping.[/yellow]")
            continue
        for _, row in df.iterrows():
            body = str(row[text_col]) if pd.notna(row[text_col]) else ""
            title = str(row[title_col]) if title_col and pd.notna(row[title_col]) else ""
            combined = (title + "\n\n" + body).strip() if title else body
            yield path, combined


def mine(args):
    console.print("[cyan]Loading StockRecognizer (regex-only, no AI)...[/cyan]")
    recognizer = StockRecognizer(use_ai=False)
    valid_tickers = set(recognizer.valid_tickers)
    # The engine only loads equities, so famous ETFs (SPY, QQQ, VOO, ...)
    # aren't in valid_tickers and would slip through as "no entity". Pull them
    # explicitly so the negative filter is strict about known tradeables.
    try:
        etfs = fd.ETFs().select()
        for t in etfs.index:
            if isinstance(t, str) and not any(ext in t for ext in EXCHANGE_BLACKLIST):
                valid_tickers.add(t)
    except Exception as exc:
        console.print(f"[yellow]Could not extend with ETFs: {exc}[/yellow]")
    # company_to_ticker keys are uppercase first-words used as company anchors
    company_first_words = set(recognizer.company_to_ticker.keys())
    console.print(
        f"  valid_tickers (incl. ETFs): {len(valid_tickers)} | "
        f"company first-words: {len(company_first_words)}"
    )

    console.print("[cyan]Indexing existing labeled + test data for dedup...[/cyan]")
    seen_hashes = collect_seen_hashes(["data/labeled", "data/test"])
    console.print(f"  Indexed {len(seen_hashes)} existing task hashes.")

    # Pass 1: collect and score every candidate.
    candidates = []
    skipped_dup = 0
    skipped_has_entity = 0
    skipped_empty = 0
    skipped_no_shape = 0

    for source, raw_text in load_sources(args.sources):
        cleaned = clean_reddit_markdown(raw_text)
        if not cleaned or len(cleaned) < 40:
            skipped_empty += 1
            continue

        h = normalize_for_dedup(cleaned)
        if h in seen_hashes:
            skipped_dup += 1
            continue

        if CASHTAG_RE.search(cleaned):
            skipped_has_entity += 1
            continue
        if has_known_ticker(cleaned, valid_tickers):
            skipped_has_entity += 1
            continue
        if likely_has_company(cleaned, company_first_words):
            skipped_has_entity += 1
            continue

        score, seeds, shaped = score_text(cleaned, valid_tickers)
        if shaped == 0:
            skipped_no_shape += 1
            continue
        if score < args.min_score:
            continue

        candidates.append({
            "source": os.path.basename(source),
            "text": cleaned,
            "score": score,
            "seeds": seeds,
            "ticker_shaped": shaped,
            "hash": h,
        })

    console.print(
        f"\n[bold]Candidates: {len(candidates)}[/bold] "
        f"(dup: {skipped_dup}, has-entity: {skipped_has_entity}, "
        f"too-short/empty: {skipped_empty}, no-ticker-shape: {skipped_no_shape})"
    )
    if not candidates:
        console.print("[red]No candidates found — try lowering --min-score or adding sources.[/red]")
        return

    # Pass 2: select top-N by score, then a diverse tail of random low-scorers.
    candidates.sort(key=lambda c: c["score"], reverse=True)
    n_top = max(1, int(args.n * (1 - args.diversity_frac)))
    n_random = args.n - n_top
    selected = candidates[:n_top]
    if n_random > 0 and len(candidates) > n_top:
        rng = random.Random(args.seed)
        tail = candidates[n_top:]
        rng.shuffle(tail)
        selected.extend(tail[:n_random])

    # Diagnostics: score distribution, top seeds.
    score_bins = {"≥6": 0, "4-6": 0, "2-4": 0, "<2": 0}
    seed_counter = {}
    for c in selected:
        s = c["score"]
        if s >= 6:
            score_bins["≥6"] += 1
        elif s >= 4:
            score_bins["4-6"] += 1
        elif s >= 2:
            score_bins["2-4"] += 1
        else:
            score_bins["<2"] += 1
        for seed in c["seeds"]:
            seed_counter[seed] = seed_counter.get(seed, 0) + 1

    bins_table = Table(title="Selected by score bucket", show_lines=False)
    bins_table.add_column("score")
    bins_table.add_column("count", justify="right")
    for k, v in score_bins.items():
        bins_table.add_row(k, str(v))
    console.print(bins_table)

    top_seeds = sorted(seed_counter.items(), key=lambda kv: kv[1], reverse=True)[:15]
    if top_seeds:
        seed_table = Table(title="Top seed FPs in selected negatives", show_lines=False)
        seed_table.add_column("seed")
        seed_table.add_column("count", justify="right")
        for seed, count in top_seeds:
            seed_table.add_row(seed, str(count))
        console.print(seed_table)

    # Preview a few.
    preview = Table(title="Sample negatives (top 5)", show_lines=False)
    preview.add_column("score", justify="right", style="dim")
    preview.add_column("seeds")
    preview.add_column("preview")
    for c in selected[:5]:
        preview.add_row(
            f"{c['score']:.1f}",
            ",".join(c["seeds"][:5]),
            c["text"][:140].replace("\n", " ") + ("..." if len(c["text"]) > 140 else ""),
        )
    console.print(preview)

    if args.dry_run:
        console.print("[yellow]--dry-run set, no file written.[/yellow]")
        return

    # Emit Label Studio tasks with empty result so the trainer treats each
    # chunk as a negative supervision sample (via the `classifications` field).
    tasks = []
    for i, c in enumerate(selected):
        tasks.append({
            "id": ID_OFFSET + i,
            "data": {"text": c["text"]},
            "annotations": [{
                "was_cancelled": False,
                "result": [],
            }],
            "_mined_meta": {
                "score": c["score"],
                "seeds": c["seeds"],
                "source": c["source"],
            },
        })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    console.print(f"[bold green]Wrote {len(tasks)} negatives to {args.output}[/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", default=[
        "data/wallstreetbets_posts.csv",
        "data/wsb.csv",
    ])
    parser.add_argument("--output", default="data/labeled/negatives_mined.json")
    parser.add_argument("--n", type=int, default=300, help="How many negatives to emit.")
    parser.add_argument("--min-score", type=float, default=1.0,
                        help="Skip candidates below this hard-negative score.")
    parser.add_argument("--diversity-frac", type=float, default=0.2,
                        help="Fraction of selections drawn randomly from below-top-N.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats but don't write output.")
    args = parser.parse_args()
    mine(args)

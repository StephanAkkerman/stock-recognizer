"""Mine financial-abbreviation negative examples from Reddit.

Scrapes posts containing PDT, SG&A, RSU, EV (enterprise value), PT (price
target), and similar abbreviations that are NOT stock tickers. Training on
these suppresses the model's tendency to fire on common financial acronyms.

Targets subreddits where financial concepts are discussed WITHOUT specific
stock picks (r/personalfinance, r/financialindependence, r/investing).
Wallstreetbets is intentionally excluded — almost every post there mentions
real tickers, leaving nothing after the entity filter.

Requires REDDIT_* env vars (same as scraper.py).

Run:
    python utils/mine_abbrev_negatives.py
    python utils/mine_abbrev_negatives.py --n 200 --dry-run
    python utils/mine_abbrev_negatives.py --output data/labeled/abbrev_negatives.json
    python utils/mine_abbrev_negatives.py --subreddits personalfinance investing
"""

import argparse
import asyncio
import json
import os
import random
import re

import financedatabase as fd
from rich.console import Console
from rich.table import Table

from stock_recognizer.constants import AMBIGUOUS_WORDS, EXCHANGE_BLACKLIST
from stock_recognizer.engine import StockRecognizer

try:
    from utils.cleaner import clean_reddit_markdown
    from utils.mine_hard_negatives import CASHTAG_RE, collect_seen_hashes, normalize_for_dedup
    from utils.scraper import RedditScraper
except ImportError:
    from cleaner import clean_reddit_markdown
    from mine_hard_negatives import CASHTAG_RE, collect_seen_hashes, normalize_for_dedup
    from scraper import RedditScraper

console = Console()

# Abbreviations to specifically teach the model NOT to fire on.
# Values are scoring weights — higher = more tempting for the model = more useful.
ABBREV_WEIGHTS = {
    "PDT": 3.0,  # pattern day trader
    "PDTR": 2.0,  # pattern day trader rule
    "EBITDA": 3.0,  # earnings metric
    "RSU": 3.0,  # restricted stock unit
    "ESPP": 2.5,  # employee stock purchase plan
    "DCF": 2.5,  # discounted cash flow
    "FCF": 2.5,  # free cash flow
    "SGA": 2.5,  # SG&A with & stripped
    "ROIC": 2.5,  # return on invested capital
    "EV": 2.0,  # enterprise value
    "PT": 2.0,  # price target
    "IV": 2.0,  # implied volatility
    "CPI": 2.0,  # consumer price index
    "PPI": 2.0,  # producer price index
    "PCE": 2.0,  # personal consumption expenditures
    "PE": 1.5,  # price-to-earnings
    "PEG": 2.0,  # price/earnings-to-growth
    "ROE": 2.0,  # return on equity
    "ROA": 2.0,  # return on assets
    "OTM": 2.0,  # out of the money
    "ITM": 2.0,  # in the money
    "ATM": 1.5,  # at the money (options)
    "GTC": 1.5,  # good till cancelled
    "ISO": 2.0,  # incentive stock option
    "TAM": 2.0,  # total addressable market
    "NPV": 2.0,  # net present value
    "IRR": 2.0,  # internal rate of return
    "EPS": 1.5,  # earnings per share
    "WACC": 2.5,  # weighted avg cost of capital
}

# Reddit search queries designed to surface abbreviation-heavy posts.
ABBREV_QUERIES = [
    "PDT rule",
    "pattern day trader",
    "SG&A expenses",
    "RSU vesting",
    "restricted stock units",
    "enterprise value EV",
    "price target analyst",
    "EBITDA margin",
    "DCF model valuation",
    "implied volatility IV",
    "CPI inflation data",
    "free cash flow FCF",
    "ROIC return on capital",
    "WACC discount rate",
    "options theta decay",
    "options gamma delta neutral",
    "earnings per share EPS",
    "discounted cash flow",
    "total addressable market TAM",
    "net present value NPV",
    "ESPP employee stock",
    "pattern day trading rules",
]

# Task IDs start here — must not overlap with mine_hard_negatives.py (9_000_000).
ID_OFFSET = 9_500_000

_TICKER_SHAPED_RE = re.compile(r"\b[A-Z]{2,6}\b")


def _original_case_tokens(text):
    """Return all 2-6 letter tokens that were already all-caps in the original text.

    The imported has_known_ticker/likely_has_company uppercase the entire text
    before scanning, which causes common English words ('it'→'IT', 'am'→'AM',
    'or'→'OR') to match obscure tickers in the 40k-symbol universe and reject
    nearly every post. By scanning the original text we only catch tokens the
    author intentionally wrote in all-caps — i.e. genuine ticker/abbreviation
    usage — leaving normal prose words untouched.
    """
    # Strip & and / first so SG&A → SGA is still a single all-caps token.
    normalized = re.sub(r"[&/]", "", text)
    return _TICKER_SHAPED_RE.findall(normalized)


def _has_known_ticker_caps_only(text, valid_tickers):
    """True if any already-uppercase token in text is a known, unambiguous ticker."""
    for tok in _original_case_tokens(text):
        if tok in valid_tickers and tok not in AMBIGUOUS_WORDS:
            return True
    return False


def _likely_has_company_caps_only(text, company_first_words):
    """True if any already-uppercase token (4+ chars) matches a company first-word."""
    for tok in _original_case_tokens(text):
        if len(tok) >= 4 and tok in company_first_words:
            return True
    return False


def score_text(text):
    """Return (score, matched_abbrevs) based on ABBREV_WEIGHTS coverage."""
    score = 0.0
    matched = set()
    for tok in _original_case_tokens(text):
        if tok in ABBREV_WEIGHTS:
            score += ABBREV_WEIGHTS[tok]
            matched.add(tok)
    return score, list(matched)


async def scrape_candidates(args, valid_tickers, company_first_words, seen_hashes):
    """Scrape one or more subreddits and return entity-free posts with nonzero abbreviation score."""
    scraper = RedditScraper()
    raw_posts = []
    per_sub_target = max(500, args.scrape_target // len(args.subreddits))
    try:
        for sub in args.subreddits:
            console.print(
                f"[cyan]Searching r/{sub} with {len(ABBREV_QUERIES)} queries "
                f"(target {per_sub_target} posts)...[/cyan]"
            )
            posts = await scraper.search_posts(
                subreddit_name=sub,
                target=per_sub_target,
                queries=ABBREV_QUERIES,
                output_file=None,
            )
            raw_posts.extend(posts)
    finally:
        await scraper.close()

    console.print(
        f"  Fetched {len(raw_posts)} raw posts across {len(args.subreddits)} subreddit(s) — filtering..."
    )

    candidates = []
    skipped = {"dup": 0, "entity": 0, "short": 0, "low_score": 0}

    for post in raw_posts:
        title = post.get("title", "") or ""
        body = post.get("text", "") or ""
        raw = (title + "\n\n" + body).strip() if title else body
        cleaned = clean_reddit_markdown(raw)

        if not cleaned or len(cleaned) < 40:
            skipped["short"] += 1
            continue

        h = normalize_for_dedup(cleaned)
        if h in seen_hashes:
            skipped["dup"] += 1
            continue
        seen_hashes.add(h)

        if CASHTAG_RE.search(cleaned):
            skipped["entity"] += 1
            continue
        if _has_known_ticker_caps_only(cleaned, valid_tickers):
            skipped["entity"] += 1
            continue
        if _likely_has_company_caps_only(cleaned, company_first_words):
            skipped["entity"] += 1
            continue

        score, matched = score_text(cleaned)
        if score < args.min_score:
            skipped["low_score"] += 1
            continue

        candidates.append({"text": cleaned, "score": score, "matched": matched})

    console.print(
        f"  Candidates: {len(candidates)} "
        f"(dup: {skipped['dup']}, entity: {skipped['entity']}, "
        f"short: {skipped['short']}, low-score: {skipped['low_score']})"
    )
    return candidates


def select_negatives(candidates, n, diversity_frac, seed):
    """Top-N by score with a random diversity tail."""
    candidates.sort(key=lambda c: c["score"], reverse=True)
    n_top = max(1, int(n * (1 - diversity_frac)))
    selected = candidates[:n_top]
    n_random = n - n_top
    if n_random > 0 and len(candidates) > n_top:
        rng = random.Random(seed)
        tail = candidates[n_top:]
        rng.shuffle(tail)
        selected.extend(tail[:n_random])
    return selected


async def main(args):
    console.print("[cyan]Loading StockRecognizer (regex only)...[/cyan]")
    recognizer = StockRecognizer(use_ai=False)
    valid_tickers = set(recognizer.valid_tickers)
    try:
        etfs = fd.ETFs().select()
        for t in etfs.index:
            if isinstance(t, str) and not any(ext in t for ext in EXCHANGE_BLACKLIST):
                valid_tickers.add(t)
    except Exception as exc:
        console.print(f"[yellow]Could not extend with ETFs: {exc}[/yellow]")

    company_first_words = set(recognizer.company_to_ticker.keys())
    console.print(
        f"  valid_tickers (incl. ETFs): {len(valid_tickers)} | "
        f"company first-words: {len(company_first_words)}"
    )

    console.print("[cyan]Indexing existing labeled + test data for dedup...[/cyan]")
    seen_hashes = collect_seen_hashes(["data/labeled", "data/test"])
    console.print(f"  Indexed {len(seen_hashes)} existing task hashes.")

    candidates = await scrape_candidates(
        args, valid_tickers, company_first_words, seen_hashes
    )
    if not candidates:
        console.print(
            "[red]No candidates found — try lowering --min-score or --scrape-target.[/red]"
        )
        return

    n_emit = min(args.n, len(candidates))
    selected = select_negatives(candidates, n_emit, args.diversity_frac, args.seed)

    # Diagnostics: which abbreviations did we actually collect?
    abbrev_counter: dict[str, int] = {}
    for c in selected:
        for tok in c["matched"]:
            abbrev_counter[tok] = abbrev_counter.get(tok, 0) + 1

    abbrev_table = Table(title=f"Abbreviations in {len(selected)} selected negatives")
    abbrev_table.add_column("abbreviation")
    abbrev_table.add_column("posts", justify="right")
    for abbrev, count in sorted(
        abbrev_counter.items(), key=lambda kv: kv[1], reverse=True
    )[:20]:
        abbrev_table.add_row(abbrev, str(count))
    console.print(abbrev_table)

    preview = Table(title="Sample negatives (top 5 by score)")
    preview.add_column("score", justify="right", style="dim")
    preview.add_column("abbrevs")
    preview.add_column("preview")
    for c in selected[:5]:
        preview.add_row(
            f"{c['score']:.1f}",
            ", ".join(sorted(c["matched"])[:6]),
            c["text"][:140].replace("\n", " ")
            + ("..." if len(c["text"]) > 140 else ""),
        )
    console.print(preview)

    if args.dry_run:
        console.print("[yellow]--dry-run set, no file written.[/yellow]")
        return

    tasks = [
        {
            "id": ID_OFFSET + i,
            "data": {"text": c["text"]},
            "annotations": [{"was_cancelled": False, "result": []}],
            "_mined_meta": {"score": c["score"], "matched": c["matched"]},
        }
        for i, c in enumerate(selected)
    ]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    console.print(
        f"[bold green]Wrote {len(tasks)} negatives → {args.output}[/bold green]"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subreddits",
        nargs="+",
        default=["personalfinance", "financialindependence", "investing"],
        help="Subreddits to search (default: personalfinance financialindependence investing). "
        "Wallstreetbets is intentionally not the default — nearly every WSB post "
        "contains a real ticker and survives the entity filter.",
    )
    parser.add_argument(
        "--output",
        default="data/labeled/abbrev_negatives.json",
        help="Output Label Studio JSON path.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Number of negatives to emit (default: 200).",
    )
    parser.add_argument(
        "--scrape-target",
        type=int,
        default=2000,
        help="Posts to scrape before filtering (default: 2000).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=2.0,
        help="Minimum abbreviation score to keep a post (default: 2.0).",
    )
    parser.add_argument(
        "--diversity-frac",
        type=float,
        default=0.2,
        help="Fraction of output drawn randomly from below-top-N (default: 0.2).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats but write nothing.",
    )
    args = parser.parse_args()
    asyncio.run(main(args))

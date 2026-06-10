import asyncio
import glob
import hashlib
import json
import os
import re

import pandas as pd
from rich.console import Console
from tqdm import tqdm

# asyncpraw and python-dotenv are only needed for live scraping. Importing them
# lazily keeps the pure helpers (submission_text, build_search_queries,
# _ScrapeState) importable — and unit-testable — in environments without the
# Reddit client installed.

console = Console()

# Lower bound stays at 50 to drop trivial one-liner *self* posts. The upper
# bound used to be 2000 to narrow the corpus toward short content, but the
# goal now is volume: we want the long DD-style posts back too, so the cap is
# raised to 10000 (still dropping pathological multi-page walls). Pass
# ``max_chars=None`` to disable the cap entirely.
MIN_CHARS_DEFAULT = 50
MAX_CHARS_DEFAULT = 10_000
# Link/image posts have no selftext. Their *title* can be the ticker-dense
# payload ("YOLO 100k $GME calls"), but using it produces rows whose text just
# duplicates the title (e.g. gains-screenshot posts), which is low-signal noise.
# So link-title capture is OFF by default (``include_link_titles=False``); when
# explicitly enabled, titles get this lower length floor instead of `min_chars`.
MIN_TITLE_CHARS_DEFAULT = 12

# Keyword queries fed to reddit search. Each returns a result set independent
# of the ~1000-item top/hot/new listing cap, so searching many terms is the
# primary way to pull *more* submissions from a subreddit we've already drained.
DEFAULT_SEARCH_KEYWORDS = [
    "$",
    "calls",
    "puts",
    "yolo",
    "dd",
    "earnings",
    "gains",
    "loss porn",
    "moon",
    "short squeeze",
    "options",
    "bagholder",
    "tendies",
    "diamond hands",
    "buy the dip",
]

# A modest curated list of perennially high-traffic WSB tickers, used to seed
# per-ticker search queries from `__main__`. Pass a larger list (e.g. an
# engine's `valid_tickers`) to `build_search_queries` to go wider.
POPULAR_TICKERS = [
    "GME", "AMC", "TSLA", "NVDA", "AAPL", "SPY", "QQQ", "AMD", "PLTR", "META",
    "MSFT", "AMZN", "GOOGL", "BABA", "NIO", "F", "BB", "NOK", "SOFI", "HOOD",
    "COIN", "RIVN", "LCID", "MARA", "RIOT", "INTC", "BAC", "DIS", "PYPL", "SNAP",
    "NFLX", "MU", "T", "WMT", "BBBY", "SPCE", "WISH", "CLOV", "TLRY", "SNDL",
]

# Sort / time-filter combinations applied to every search query. "relevance"
# and "top" honour `time_filter`; "new" ignores it but surfaces a different
# (recency-ordered) slice, maximising coverage per query.
DEFAULT_SEARCH_SORTS = [("relevance", "all"), ("new", "all"), ("top", "all")]

# Time filters for the time-aware listings (top, controversial).
DEFAULT_TIME_FILTERS = ["week", "month", "year", "all"]


def _text_dedup_hash(text):
    """SHA1 of the first 200 chars of normalised text — cheap fuzzy dedup."""
    if not text:
        return None
    norm = re.sub(r"\s+", " ", text.strip().lower())[:200]
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def submission_text(
    submission,
    min_chars=MIN_CHARS_DEFAULT,
    max_chars=MAX_CHARS_DEFAULT,
    include_link_titles=False,
    min_title_chars=MIN_TITLE_CHARS_DEFAULT,
):
    """Pick the trainable text out of a submission and length-filter it.

    Self posts contribute their ``selftext`` (floored at ``min_chars``). Link /
    image posts have none, so — when ``include_link_titles`` is set — their
    ``title`` is used instead, floored at the lower ``min_title_chars``.

    Returns
    -------
    tuple[str | None, str | None]
        ``(text, None)`` when the submission is kept, otherwise ``(None,
        reason)`` where ``reason`` is one of ``"too_short"``, ``"too_long"`` or
        ``"link_post"``.
    """
    is_self = getattr(submission, "is_self", True)
    if is_self:
        text = getattr(submission, "selftext", "") or ""
        floor = min_chars
    else:
        if not include_link_titles:
            return None, "link_post"
        text = getattr(submission, "title", "") or ""
        floor = min_title_chars

    if len(text) < floor:
        return None, "too_short"
    if max_chars is not None and len(text) > max_chars:
        return None, "too_long"
    return text, None


def build_search_queries(keywords=None, tickers=None):
    """Build a deduplicated, order-preserving list of reddit search queries.

    ``keywords`` defaults to :data:`DEFAULT_SEARCH_KEYWORDS`; ``tickers`` are
    appended verbatim. Duplicates are removed case-insensitively (first spelling
    wins) so ``["DD", "dd"]`` collapses to a single query.
    """
    keywords = DEFAULT_SEARCH_KEYWORDS if keywords is None else keywords
    tickers = tickers or []
    queries = []
    seen = set()
    for q in list(keywords) + list(tickers):
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(q)
    return queries


def load_known_post_ids(csv_paths):
    """Reddit submission IDs from any prior scrape CSVs in `csv_paths`."""
    known = set()
    for path in csv_paths:
        if not os.path.exists(path):
            continue

        try:
            df = pd.read_csv(path, usecols=["id"])
            known.update(str(i) for i in df["id"].dropna())
            break
        except (ValueError, KeyError):
            continue
    return known


def load_known_text_hashes_from_csvs(csv_paths, row_limit: int = 100_000):
    """Text-prefix hashes from CSV files that have a ``text`` column.

    Skips files with more than ``row_limit`` rows to avoid slow hashing of
    multi-million-row datasets (use ID-based dedup for those instead).
    """
    hashes = set()
    for path in csv_paths:
        if not os.path.exists(path):
            continue
        try:
            row_count = sum(1 for _ in open(path, encoding="utf-8")) - 1
            if row_count > row_limit:
                console.print(
                    f"[dim]Skipping text-hash dedup for {path} "
                    f"({row_count:,} rows > limit {row_limit:,})[/dim]"
                )
                continue
            df = pd.read_csv(path, usecols=["text"])
            for text in df["text"].dropna():
                h = _text_dedup_hash(str(text))
                if h:
                    hashes.add(h)
        except (ValueError, KeyError, pd.errors.EmptyDataError, OSError):
            continue
    return hashes


def load_known_text_hashes(label_folders):
    """Text-prefix hashes from labeled+test JSON so we don't re-scrape posts
    that already exist in annotated form (where the Reddit ID was lost in the
    labeling pipeline)."""
    hashes = set()
    for folder in label_folders:
        if not os.path.isdir(folder):
            continue
        for fp in glob.glob(os.path.join(folder, "*.json")):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(data, list):
                continue
            for task in data:
                text = (
                    task.get("data", {}).get("text") if isinstance(task, dict) else None
                )
                h = _text_dedup_hash(text)
                if h:
                    hashes.add(h)
    return hashes


def _append_chunk(output_file, chunk, prior_count):
    """Append `chunk` to `output_file`, writing the header only on first write
    to a file that wasn't already present (so re-running mid-CSV doesn't
    duplicate the header row)."""
    write_header = prior_count == 0 and not os.path.exists(output_file)
    pd.DataFrame(chunk).to_csv(output_file, mode="a", header=write_header, index=False)


class _ScrapeState:
    """Mutable accumulator shared across every source in a single scrape.

    Holds the dedup universe, collected posts, skip counters and incremental
    CSV save bookkeeping. :meth:`add` is the single chokepoint every fetched
    submission flows through, which keeps id/text dedup, length filtering and
    saving consistent whether the submission came from a listing or a search.
    """

    def __init__(
        self,
        known_ids,
        known_text_hashes,
        output_file=None,
        save_every=50,
        min_chars=MIN_CHARS_DEFAULT,
        max_chars=MAX_CHARS_DEFAULT,
        include_link_titles=False,
        min_title_chars=MIN_TITLE_CHARS_DEFAULT,
    ):
        self.posts = []
        self.seen_ids = set(known_ids)
        self.known_text_hashes = set(known_text_hashes)
        self.output_file = output_file
        self.save_every = save_every
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.include_link_titles = include_link_titles
        self.min_title_chars = min_title_chars
        self.last_saved = 0
        self.pbar = None
        self.counters = {
            "dup_id": 0,
            "dup_text": 0,
            "too_short": 0,
            "too_long": 0,
            "link_post": 0,
        }

    def add(self, submission):
        """Filter, dedup and (optionally) persist a single submission.

        Returns ``True`` if it was appended, ``False`` if skipped (with the
        relevant counter incremented)."""
        sid = getattr(submission, "id", None)
        if sid is not None and sid in self.seen_ids:
            self.counters["dup_id"] += 1
            return False
        if sid is not None:
            self.seen_ids.add(sid)

        text, reason = submission_text(
            submission,
            self.min_chars,
            self.max_chars,
            self.include_link_titles,
            self.min_title_chars,
        )
        if reason is not None:
            self.counters[reason] += 1
            return False

        text_hash = _text_dedup_hash(text)
        if text_hash in self.known_text_hashes:
            self.counters["dup_text"] += 1
            return False
        self.known_text_hashes.add(text_hash)

        self.posts.append(
            {
                "id": sid,
                "title": getattr(submission, "title", ""),
                "text": text,
                "score": getattr(submission, "score", None),
                "url": (
                    f"https://reddit.com{submission.permalink}"
                    if getattr(submission, "permalink", None)
                    else getattr(submission, "url", None)
                ),
                "created_utc": getattr(submission, "created_utc", None),
            }
        )
        if self.pbar is not None:
            self.pbar.update(1)

        if self.output_file and len(self.posts) - self.last_saved >= self.save_every:
            _append_chunk(self.output_file, self.posts[self.last_saved :], self.last_saved)
            self.last_saved = len(self.posts)
        return True

    def flush(self):
        """Persist any posts collected since the last incremental save."""
        if self.output_file and len(self.posts) > self.last_saved:
            _append_chunk(self.output_file, self.posts[self.last_saved :], self.last_saved)
            self.last_saved = len(self.posts)


class RedditScraper:
    def __init__(self):
        from dotenv import load_dotenv

        load_dotenv()
        self.reddit = None

    async def _init_reddit(self):
        if self.reddit is None:
            import asyncpraw

            self.reddit = asyncpraw.Reddit(
                client_id=os.getenv("REDDIT_PERSONAL_USE"),
                client_secret=os.getenv("REDDIT_SECRET"),
                user_agent=os.getenv("REDDIT_APP_NAME"),
                username=os.getenv("REDDIT_USERNAME"),
                password=os.getenv("REDDIT_PASSWORD"),
            )

    async def _preflight(self, subreddit_name):
        """Fail fast if auth or the subreddit handle is broken — surface that
        before users sit through a noop scrape and discover an empty CSV."""
        await self._init_reddit()
        subreddit = await self.reddit.subreddit(subreddit_name)
        async for _ in subreddit.top(time_filter="day", limit=1):
            return subreddit
        raise RuntimeError(
            f"Preflight failed: r/{subreddit_name} returned 0 posts under "
            f"top(day, limit=1). Check REDDIT_* env vars and subreddit name."
        )

    def _make_state(
        self,
        output_file,
        save_every,
        dedup_csvs,
        dedup_text_folders,
        min_chars,
        max_chars,
        include_link_titles,
        min_title_chars,
    ):
        """Build the dedup universe and return a fresh :class:`_ScrapeState`."""
        dedup_csvs = dedup_csvs or ([output_file] if output_file else [])
        dedup_csvs = [p for p in dedup_csvs if p]
        dedup_text_folders = dedup_text_folders or ["data/labeled", "data/test"]

        known_ids = load_known_post_ids(dedup_csvs)
        known_text_hashes = load_known_text_hashes(dedup_text_folders)
        known_text_hashes |= load_known_text_hashes_from_csvs(dedup_csvs)
        console.print(
            f"[cyan]Dedup universe: {len(known_ids)} prior Reddit IDs, "
            f"{len(known_text_hashes)} text hashes[/cyan]"
        )
        return _ScrapeState(
            known_ids=known_ids,
            known_text_hashes=known_text_hashes,
            output_file=output_file,
            save_every=save_every,
            min_chars=min_chars,
            max_chars=max_chars,
            include_link_titles=include_link_titles,
            min_title_chars=min_title_chars,
        )

    @staticmethod
    def _listing_sources(subreddit, time_filters=None):
        """All listing endpoints as ``(label, async_iter)`` pairs.

        `top` and `controversial` are drawn across every time filter (each a
        separate ~1000-item window); `hot`, `new` and `rising` add three more
        windows that are largely disjoint from the time-ranked ones.
        """
        time_filters = time_filters or DEFAULT_TIME_FILTERS
        sources = []
        for tf in time_filters:
            sources.append((f"top/{tf}", subreddit.top(time_filter=tf, limit=None)))
            sources.append(
                (
                    f"controversial/{tf}",
                    subreddit.controversial(time_filter=tf, limit=None),
                )
            )
        sources.append(("hot", subreddit.hot(limit=None)))
        sources.append(("new", subreddit.new(limit=None)))
        sources.append(("rising", subreddit.rising(limit=None)))
        return sources

    @staticmethod
    def _search_sources(subreddit, queries, sorts=None):
        """Reddit search as ``(label, async_iter)`` pairs — one per
        (query, sort/time_filter) combination."""
        sorts = sorts or DEFAULT_SEARCH_SORTS
        sources = []
        for q in queries:
            for sort, tf in sorts:
                sources.append(
                    (
                        f"search:{q}:{sort}",
                        subreddit.search(q, sort=sort, time_filter=tf, limit=None),
                    )
                )
        return sources

    async def _collect(self, sources, state, target, desc):
        """Drain ``sources`` into ``state`` until ``target`` posts are kept."""
        with tqdm(total=target, desc=desc) as pbar:
            state.pbar = pbar
            pbar.update(len(state.posts))  # reflect any pre-collected posts
            for label, aiter in sources:
                if len(state.posts) >= target:
                    break
                pbar.set_postfix_str(label)
                async for submission in aiter:
                    state.add(submission)
                    if len(state.posts) >= target:
                        break
        state.pbar = None
        state.flush()
        return state.posts

    async def fetch_posts(
        self,
        subreddit_name: str,
        target: int = 200,
        min_chars: int = MIN_CHARS_DEFAULT,
        max_chars: int = MAX_CHARS_DEFAULT,
        output_file: str = None,
        save_every: int = 50,
        dedup_csvs: list[str] = None,
        dedup_text_folders: list[str] = None,
        include_link_titles: bool = False,
        time_filters: list[str] = None,
    ) -> list[dict]:
        """Scrape submissions from every listing endpoint of a subreddit."""
        subreddit = await self._preflight(subreddit_name)
        state = self._make_state(
            output_file,
            save_every,
            dedup_csvs,
            dedup_text_folders,
            min_chars,
            max_chars,
            include_link_titles,
            MIN_TITLE_CHARS_DEFAULT,
        )
        sources = self._listing_sources(subreddit, time_filters)
        posts = await self._collect(
            sources, state, target, f"Scraping r/{subreddit_name}"
        )
        self._report(posts, target, output_file, state.counters)
        return posts

    async def search_posts(
        self,
        subreddit_name: str,
        target: int = 500,
        queries: list[str] = None,
        tickers: list[str] = None,
        min_chars: int = MIN_CHARS_DEFAULT,
        max_chars: int = MAX_CHARS_DEFAULT,
        output_file: str = None,
        save_every: int = 50,
        dedup_csvs: list[str] = None,
        dedup_text_folders: list[str] = None,
        include_link_titles: bool = False,
        sorts: list[tuple] = None,
    ) -> list[dict]:
        """Scrape submissions via reddit full-text search.

        Each query (keywords from :func:`build_search_queries` plus any
        ``tickers``) returns a result set independent of the listing cap, so
        this reaches submissions the top/hot/new windows never expose.
        """
        subreddit = await self._preflight(subreddit_name)
        if queries is None:
            queries = build_search_queries(tickers=tickers)
        state = self._make_state(
            output_file,
            save_every,
            dedup_csvs,
            dedup_text_folders,
            min_chars,
            max_chars,
            include_link_titles,
            MIN_TITLE_CHARS_DEFAULT,
        )
        sources = self._search_sources(subreddit, queries, sorts)
        posts = await self._collect(
            sources, state, target, f"Searching r/{subreddit_name}"
        )
        self._report(posts, target, output_file, state.counters)
        return posts

    async def harvest(
        self,
        subreddit_name: str,
        target: int = 1000,
        queries: list[str] = None,
        tickers: list[str] = None,
        min_chars: int = MIN_CHARS_DEFAULT,
        max_chars: int = MAX_CHARS_DEFAULT,
        output_file: str = None,
        save_every: int = 50,
        dedup_csvs: list[str] = None,
        dedup_text_folders: list[str] = None,
        include_link_titles: bool = False,
        time_filters: list[str] = None,
        sorts: list[tuple] = None,
    ) -> list[dict]:
        """Listings *and* search in one pass, sharing a single dedup state.

        Listings run first (cheap, high-signal), then search top-ups the rest of
        ``target``. Because both share one :class:`_ScrapeState`, a submission
        seen in a listing is never re-added by a later search.
        """
        subreddit = await self._preflight(subreddit_name)
        if queries is None:
            queries = build_search_queries(tickers=tickers)
        state = self._make_state(
            output_file,
            save_every,
            dedup_csvs,
            dedup_text_folders,
            min_chars,
            max_chars,
            include_link_titles,
            MIN_TITLE_CHARS_DEFAULT,
        )
        sources = self._listing_sources(subreddit, time_filters) + self._search_sources(
            subreddit, queries, sorts
        )
        posts = await self._collect(
            sources, state, target, f"Harvesting r/{subreddit_name}"
        )
        self._report(posts, target, output_file, state.counters)
        return posts

    @staticmethod
    def _report(posts, target, output_file, skip_counts):
        """Print a clear success/partial/failure summary at the end of a run."""
        total_skipped = sum(skip_counts.values())
        if posts:
            status = "SUCCESS" if len(posts) >= target else "PARTIAL"
            colour = "green" if status == "SUCCESS" else "yellow"
        else:
            status = "FAILURE"
            colour = "red"

        console.print(
            f"\n[bold {colour}]{status}[/bold {colour}] — collected "
            f"[bold]{len(posts)}[/bold] / {target} new posts"
        )
        console.print(
            f"  skipped: dup_id={skip_counts['dup_id']} "
            f"dup_text={skip_counts['dup_text']} "
            f"too_short={skip_counts['too_short']} "
            f"too_long={skip_counts['too_long']} "
            f"link_post={skip_counts['link_post']} "
            f"(total {total_skipped})"
        )

        if output_file:
            if os.path.exists(output_file):
                try:
                    n_rows = len(pd.read_csv(output_file))
                    console.print(f"  CSV: {output_file} ({n_rows} total rows on disk)")
                except Exception as exc:
                    console.print(f"[red]  CSV read-back failed: {exc}[/red]")
            else:
                console.print(f"[red]  CSV not written: {output_file} missing[/red]")

    async def close(self):
        if self.reddit is not None:
            await self.reddit.close()


if __name__ == "__main__":
    OUTPUT = "data/wallstreetbets_posts.csv"
    # All prior CSV sources — used for both ID dedup and (where small enough)
    # text-hash dedup so we never re-scrape posts already in the dataset.
    ALL_CSVS = [
        OUTPUT,
        "data/wsb.csv",
    ]
    queries = build_search_queries(tickers=POPULAR_TICKERS)
    scraper = RedditScraper()
    posts = asyncio.run(
        scraper.harvest(
            "wallstreetbets",
            target=2000,
            queries=queries,
            output_file=OUTPUT,
            dedup_csvs=ALL_CSVS,
        )
    )
    asyncio.run(scraper.close())

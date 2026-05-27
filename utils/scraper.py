import asyncio
import glob
import hashlib
import json
import os
import re

import asyncpraw
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from tqdm import tqdm

console = Console()

# Reasonable bounds that match the working training distribution. The previous
# `min_chars=100` plus no cap pulled the corpus toward long DD-style posts;
# the held-out test set averages ~254 chars/post and the bulk of WSB content
# people quote in conversation is also short. 50 < len <= 2000 captures both
# cashtag-shouting short posts and medium-length analyses while dropping
# 2000+ char mega-DDs that overrepresent labeled_final's distribution.
MIN_CHARS_DEFAULT = 50
MAX_CHARS_DEFAULT = 2000


def _text_dedup_hash(text):
    """SHA1 of the first 200 chars of normalised text — cheap fuzzy dedup."""
    if not text:
        return None
    norm = re.sub(r"\s+", " ", text.strip().lower())[:200]
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def load_known_post_ids(csv_paths):
    """Reddit submission IDs from any prior scrape CSVs in `csv_paths`."""
    known = set()
    for path in csv_paths:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, usecols=["id"])
            known.update(str(i) for i in df["id"].dropna())
        except (ValueError, KeyError, pd.errors.EmptyDataError):
            continue
    return known


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
                text = task.get("data", {}).get("text") if isinstance(task, dict) else None
                h = _text_dedup_hash(text)
                if h:
                    hashes.add(h)
    return hashes


class RedditScraper:
    def __init__(self):
        load_dotenv()
        self.reddit = None

    async def _init_reddit(self):
        if self.reddit is None:
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
    ) -> list[dict]:
        subreddit = await self._preflight(subreddit_name)

        # Build the dedup universe before scraping starts.
        dedup_csvs = dedup_csvs or ([output_file] if output_file else [])
        dedup_csvs = [p for p in dedup_csvs if p]
        dedup_text_folders = dedup_text_folders or ["data/labeled", "data/test"]

        known_ids = load_known_post_ids(dedup_csvs)
        known_text_hashes = load_known_text_hashes(dedup_text_folders)
        console.print(
            f"[cyan]Dedup universe: {len(known_ids)} prior Reddit IDs, "
            f"{len(known_text_hashes)} labeled-text hashes[/cyan]"
        )

        posts = []
        seen_ids: set[str] = set(known_ids)
        last_saved = 0
        skipped_dup_id = 0
        skipped_dup_text = 0
        skipped_too_short = 0
        skipped_too_long = 0
        skipped_link_post = 0
        time_filters = ["week", "month", "year", "all"]

        with tqdm(total=target, desc=f"Scraping r/{subreddit_name}") as pbar:
            for time_filter in time_filters:
                if len(posts) >= target:
                    break

                pbar.set_postfix(filter=time_filter)
                async for submission in subreddit.top(time_filter=time_filter, limit=None):
                    if submission.id in seen_ids:
                        skipped_dup_id += 1
                        continue
                    seen_ids.add(submission.id)

                    if not submission.is_self:
                        skipped_link_post += 1
                        continue

                    text = submission.selftext or ""
                    if len(text) < min_chars:
                        skipped_too_short += 1
                        continue
                    if len(text) > max_chars:
                        skipped_too_long += 1
                        continue

                    text_hash = _text_dedup_hash(text)
                    if text_hash in known_text_hashes:
                        skipped_dup_text += 1
                        continue
                    known_text_hashes.add(text_hash)

                    posts.append(
                        {
                            "id": submission.id,
                            "title": submission.title,
                            "text": text,
                            "score": submission.score,
                            "url": f"https://reddit.com{submission.permalink}",
                            "created_utc": submission.created_utc,
                        }
                    )
                    pbar.update(1)

                    if output_file and len(posts) - last_saved >= save_every:
                        self._append_chunk(output_file, posts[last_saved:], last_saved)
                        last_saved = len(posts)

                    if len(posts) >= target:
                        break

        if output_file and len(posts) > last_saved:
            self._append_chunk(output_file, posts[last_saved:], last_saved)

        self._report(
            posts,
            target,
            output_file,
            {
                "dup_id": skipped_dup_id,
                "dup_text": skipped_dup_text,
                "too_short": skipped_too_short,
                "too_long": skipped_too_long,
                "link_post": skipped_link_post,
            },
        )

        return posts

    @staticmethod
    def _append_chunk(output_file, chunk, prior_count):
        """Append `chunk` to `output_file`, writing the header only on first write
        to a file that wasn't already present (so re-running mid-CSV doesn't
        duplicate the header row)."""
        write_header = prior_count == 0 and not os.path.exists(output_file)
        pd.DataFrame(chunk).to_csv(
            output_file, mode="a", header=write_header, index=False
        )

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
                    console.print(
                        f"  CSV: {output_file} ({n_rows} total rows on disk)"
                    )
                except Exception as exc:
                    console.print(f"[red]  CSV read-back failed: {exc}[/red]")
            else:
                console.print(f"[red]  CSV not written: {output_file} missing[/red]")

    async def close(self):
        if self.reddit is not None:
            await self.reddit.close()


if __name__ == "__main__":
    OUTPUT = "data/wallstreetbets_posts.csv"
    scraper = RedditScraper()
    posts = asyncio.run(
        scraper.fetch_posts(
            "wallstreetbets",
            target=500,
            output_file=OUTPUT,
        )
    )
    asyncio.run(scraper.close())

import os

import asyncpraw
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm


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

    async def fetch_posts(
        self,
        subreddit_name: str,
        target: int = 200,
        min_chars: int = 100,
        output_file: str = None,
        save_every: int = 50,
    ) -> list[dict]:
        await self._init_reddit()

        subreddit = await self.reddit.subreddit(subreddit_name)
        posts = []
        seen_ids: set[str] = set()
        last_saved = 0
        time_filters = ["week", "month", "year", "all"]

        with tqdm(total=target, desc=f"Scraping r/{subreddit_name}") as pbar:
            for time_filter in time_filters:
                if len(posts) >= target:
                    break

                pbar.set_postfix(filter=time_filter)
                async for submission in subreddit.top(time_filter=time_filter, limit=None):
                    if submission.id in seen_ids:
                        continue
                    seen_ids.add(submission.id)

                    if not submission.is_self:
                        continue

                    if len(submission.selftext) < min_chars:
                        continue

                    posts.append(
                        {
                            "id": submission.id,
                            "title": submission.title,
                            "text": submission.selftext,
                            "score": submission.score,
                            "url": f"https://reddit.com{submission.permalink}",
                            "created_utc": submission.created_utc,
                        }
                    )
                    pbar.update(1)

                    if output_file and len(posts) - last_saved >= save_every:
                        chunk = posts[last_saved:]
                        pd.DataFrame(chunk).to_csv(
                            output_file, mode="a", header=(last_saved == 0), index=False
                        )
                        last_saved = len(posts)

                    if len(posts) >= target:
                        break

        if output_file and len(posts) > last_saved:
            chunk = posts[last_saved:]
            pd.DataFrame(chunk).to_csv(
                output_file, mode="a", header=(last_saved == 0), index=False
            )

        return posts

    async def close(self):
        if self.reddit is not None:
            await self.reddit.close()


if __name__ == "__main__":
    import asyncio

    OUTPUT = "wallstreetbets_posts.csv"
    scraper = RedditScraper()
    posts = asyncio.run(
        scraper.fetch_posts("wallstreetbets", target=200, output_file=OUTPUT)
    )
    asyncio.run(scraper.close())

    print(f"Scraped {len(posts)} posts → {OUTPUT}")

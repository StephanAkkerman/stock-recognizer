import asyncio
from datetime import datetime

import asyncpraw
import pandas as pd


async def fetch_reddit_data(limit=500, min_chars=300):
    reddit = asyncpraw.Reddit(
        client_id="YOUR_ID",
        client_secret="YOUR_SECRET",
        user_agent="StockRecognizer_v2_Scraper",
    )

    subreddit = await reddit.subreddit("wallstreetbets")
    posts = []

    # Using 'top' with 'year' to get high-engagement posts
    async for submission in subreddit.top(time_filter="year", limit=limit):
        # 1. Filter: Only Self-posts (no direct image/link posts)
        if not submission.is_self:
            continue

        # 2. Filter: 2026 only
        post_date = datetime.fromtimestamp(submission.created_utc)
        if post_date.year < 2026:
            continue

        # 3. Filter: Character count
        if len(submission.selftext) < min_chars:
            continue

        posts.append(
            {
                "id": submission.id,
                "title": submission.title,
                "text": submission.selftext,
                "score": submission.score,
                "url": submission.url,
                "created_utc": submission.created_utc,
            }
        )

    await reddit.close()
    return pd.DataFrame(posts)


# Run it
# df = asyncio.run(fetch_reddit_data())

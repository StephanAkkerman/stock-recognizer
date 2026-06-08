"""Cleans the kaggle_wsb.csv to only keep posts that are likely to be good for training.
The dataset originally includes comments, so we drop those by only keeping posts with a non-empty title.
"""

import pandas


def main():
    df = pandas.read_csv("data/kaggle_wsb.csv")
    print(f"original dataset: {len(df)} rows")

    # Only keep posts with a non-empty title, which are likely to be the original posts and not comments
    df = df[df["title"].notna() & (df["title"].str.strip() != "")]
    print(f"after dropping comments: {len(df)} rows")

    df.to_csv("data/kaggle_wsb_clean.csv", index=False)
    print("wrote cleaned dataset to data/kaggle_wsb_clean.csv")


if __name__ == "__main__":
    main()

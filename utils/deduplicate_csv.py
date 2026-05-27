import pandas as pd
from pathlib import Path

csv_path = Path(__file__).parent.parent / "data" / "wallstreetbets_posts.csv"

df = pd.read_csv(csv_path)

print(f"Original rows: {len(df)}")
print(f"Unique IDs: {df['id'].nunique()}")

# Check duplicate counts
duplicates = df[df.duplicated(subset=['id'], keep=False)].sort_values('id')
if len(duplicates) > 0:
    print(f"\nDuplicate count by ID (top 10):")
    dup_counts = df[df.duplicated(subset=['id'], keep=False)]['id'].value_counts()
    print(dup_counts.head(10))

# Keep only first occurrence of each ID
df_deduplicated = df.drop_duplicates(subset=['id'], keep='first')

print(f"\nAfter deduplication:")
print(f"Rows: {len(df_deduplicated)}")
print(f"Rows removed: {len(df) - len(df_deduplicated)}")

# Write back to CSV
df_deduplicated.to_csv(csv_path, index=False)
print(f"\nSaved deduplicated CSV to {csv_path}")

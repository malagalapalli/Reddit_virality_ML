from datasets import load_dataset
import pandas as pd
import os

print("Starting download... this may take 10-20 minutes")
os.makedirs("data/raw", exist_ok=True)

# ── Attempt 1: Load via HF auto-converted parquet (bypasses dead SocialGrep server) ──
try:
    print("Trying HF parquet mirror for posts...")
    ds_posts = load_dataset(
        "SocialGrep/ten-million-reddit-answers",
        "posts",
        revision="refs/convert/parquet",
    )
    print("Converting posts to pandas...")
    df_posts = ds_posts["train"].to_pandas()

except Exception as e1:
    print(f"Parquet mirror failed ({e1}), trying direct download...")
    # ── Attempt 2: direct parquet URL on HF Hub ──
    try:
        url = ("https://huggingface.co/datasets/SocialGrep/"
               "ten-million-reddit-answers/resolve/refs%2Fconvert%2Fparquet/"
               "posts/train/0000.parquet")
        df_posts = pd.read_parquet(url)
    except Exception as e2:
        print(f"Direct parquet also failed ({e2}).")
        print("Falling back to alternative Reddit dataset...")
        # ── Attempt 3: use a different, reliable Reddit dataset ──
        ds = load_dataset("webis/tldr-17", split="train")
        df_posts = ds.to_pandas()

print(f"Posts: {len(df_posts):,} rows, {len(df_posts.columns)} columns")
print(df_posts.head())
df_posts.to_csv("data/raw/reddit_posts.csv", index=False)
print("Saved to data/raw/reddit_posts.csv")

# ── Comments (same strategy) ──
try:
    print("\nTrying HF parquet mirror for comments...")
    ds_comments = load_dataset(
        "SocialGrep/ten-million-reddit-answers",
        "comments",
        revision="refs/convert/parquet",
    )
    df_comments = ds_comments["train"].to_pandas()

except Exception as e1:
    print(f"Parquet mirror failed ({e1}), trying direct download...")
    try:
        url = ("https://huggingface.co/datasets/SocialGrep/"
               "ten-million-reddit-answers/resolve/refs%2Fconvert%2Fparquet/"
               "comments/train/0000.parquet")
        df_comments = pd.read_parquet(url)
    except Exception as e2:
        print(f"Comments download failed ({e2}). Skipping comments.")
        df_comments = None

if df_comments is not None:
    print(f"Comments: {len(df_comments):,} rows, {len(df_comments.columns)} columns")
    print(df_comments.head())
    df_comments.to_csv("data/raw/reddit_comments.csv", index=False)
    print("Saved to data/raw/reddit_comments.csv")
print("Saved to data/raw/reddit_comments.csv")

print("DONE")

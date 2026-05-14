"""
preprocess.py
-------------
Loads the raw MovieLens 100K ratings, filters, converts to implicit feedback,
remaps IDs, and splits into train/test per user.

Usage:
    python 01_data/preprocess.py

Inputs:
    data/raw/ml-100k/u.data

Outputs:
    data/processed/train.csv        <- user_id, item_id (implicit positives)
    data/processed/test.csv         <- user_id, item_id (held-out per user)
    data/processed/user_map.json    <- {original_uid: new_int_id}
    data/processed/item_map.json    <- {original_iid: new_int_id}
    data/processed/stats.json       <- dataset statistics
"""

# !! MUST be first — fixes macOS OpenMP crash before any library loads.
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import random

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def log(msg):
    print(f"[preprocess] {msg}", flush=True)


def load_raw_ratings(ratings_path):
    """
    Loads u.data (tab-separated: user_id, item_id, rating, timestamp).
    Returns a DataFrame with columns: user_id, item_id, rating, timestamp.
    Raises a clear error if the file is missing or malformed.
    """
    log(f"Loading raw ratings from: {ratings_path}")

    if not os.path.isfile(ratings_path):
        raise FileNotFoundError(
            f"[ERROR] Ratings file not found: {ratings_path}\n"
            "       Run 'python 01_data/download_movielens.py' first."
        )

    df = pd.read_csv(
        ratings_path,
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": int, "item_id": int, "rating": int, "timestamp": int},
    )

    # Sanity check: expected columns exist and values are in sane range
    assert df["rating"].between(1, 5).all(), \
        "[ERROR] Ratings should be integers 1-5. Data may be corrupted."
    assert df["user_id"].min() >= 1, \
        "[ERROR] user_id should start from 1 in raw data."

    log(f"Raw data loaded:  {len(df):,} rows, "
        f"{df['user_id'].nunique()} users, "
        f"{df['item_id'].nunique()} items")
    return df


def filter_by_frequency(df):
    """
    Removes users with fewer than MIN_USER_RATINGS interactions and
    items with fewer than MIN_ITEM_RATINGS interactions.
    Applies iteratively until stable (some users may drop after item removal).
    """
    log(f"Filtering: keep users >= {config.MIN_USER_RATINGS} ratings, "
        f"items >= {config.MIN_ITEM_RATINGS} ratings")

    before_users = df["user_id"].nunique()
    before_items = df["item_id"].nunique()
    before_rows  = len(df)

    # Iterate until no more rows are dropped (usually 2 passes is enough)
    for iteration in range(10):
        prev_len = len(df)

        # Drop low-frequency items
        item_counts = df["item_id"].value_counts()
        valid_items = item_counts[item_counts >= config.MIN_ITEM_RATINGS].index
        df = df[df["item_id"].isin(valid_items)]

        # Drop low-frequency users
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= config.MIN_USER_RATINGS].index
        df = df[df["user_id"].isin(valid_users)]

        if len(df) == prev_len:
            log(f"  Stable after {iteration + 1} iteration(s).")
            break

    log(f"After filtering: {len(df):,} rows "
        f"(removed {before_rows - len(df):,} rows, "
        f"{before_users - df['user_id'].nunique()} users, "
        f"{before_items - df['item_id'].nunique()} items)")
    log(f"Remaining: {df['user_id'].nunique()} users, "
        f"{df['item_id'].nunique()} items")
    return df


def to_implicit(df):
    """
    Converts ratings to implicit feedback.
    We treat ANY rating as a positive interaction (user watched/rated = interest).
    Drops the rating and timestamp columns — we only need (user_id, item_id).
    """
    log("Converting to implicit feedback (any rating = 1 positive interaction)")
    df = df[["user_id", "item_id"]].drop_duplicates()
    log(f"Implicit interactions: {len(df):,} unique (user, item) pairs")
    return df


def remap_ids(df):
    """
    Remaps user_id and item_id to contiguous integers starting from 0.
    This is required by PyTorch Geometric — node indices must be 0-based.

    Returns:
        df          - DataFrame with remapped IDs
        user_map    - dict {original_uid (str): new_int_id}
        item_map    - dict {original_iid (str): new_int_id}
    """
    log("Remapping user and item IDs to 0-based contiguous integers")

    unique_users = sorted(df["user_id"].unique())
    unique_items = sorted(df["item_id"].unique())

    user_map = {int(u): idx for idx, u in enumerate(unique_users)}
    item_map = {int(i): idx for idx, i in enumerate(unique_items)}

    df = df.copy()
    df["user_id"] = df["user_id"].map(user_map)
    df["item_id"] = df["item_id"].map(item_map)

    # Verify no NaN after remapping (would mean some IDs were missed)
    assert df["user_id"].isna().sum() == 0, "[ERROR] Some user_ids failed to remap."
    assert df["item_id"].isna().sum() == 0, "[ERROR] Some item_ids failed to remap."

    log(f"  Users remapped: 0 to {len(user_map) - 1}  ({len(user_map)} total)")
    log(f"  Items remapped: 0 to {len(item_map) - 1}  ({len(item_map)} total)")

    return df, user_map, item_map


def train_test_split_per_user(df):
    """
    Splits interactions into train/test per user.
    Strategy: for each user, sort by implicit order (no timestamp — random shuffle
    with fixed seed), take last 20% as test.
    This ensures every user has at least some test interactions.

    Returns:
        train_df, test_df
    """
    log(f"Splitting train/test per user (train={config.TRAIN_RATIO}, "
        f"test={config.TEST_RATIO}, seed={config.SEED})")

    rng = random.Random(config.SEED)
    train_rows = []
    test_rows  = []
    users_with_no_test = 0

    for user_id, group in df.groupby("user_id"):
        items = group["item_id"].tolist()
        rng.shuffle(items)  # reproducible shuffle

        n_test = max(1, int(len(items) * config.TEST_RATIO))  # at least 1 test item
        test_items  = items[:n_test]
        train_items = items[n_test:]

        if len(train_items) == 0:
            # Edge case: user has too few items, keep all in train
            train_items = test_items
            test_items  = []
            users_with_no_test += 1

        for item in train_items:
            train_rows.append({"user_id": user_id, "item_id": item})
        for item in test_items:
            test_rows.append({"user_id": user_id, "item_id": item})

    train_df = pd.DataFrame(train_rows)
    test_df  = pd.DataFrame(test_rows)

    if users_with_no_test > 0:
        log(f"  WARNING: {users_with_no_test} users had too few items for a test split — kept all in train.")

    log(f"  Train: {len(train_df):,} interactions")
    log(f"  Test:  {len(test_df):,} interactions")
    log(f"  Train users: {train_df['user_id'].nunique()}, "
        f"Test users: {test_df['user_id'].nunique()}")

    return train_df, test_df


def save_outputs(train_df, test_df, user_map, item_map):
    """
    Saves all processed outputs to data/processed/.
    Creates the directory if it does not exist.
    """
    os.makedirs(config.PROC_DATA_DIR, exist_ok=True)
    log(f"Saving outputs to: {config.PROC_DATA_DIR}")

    # Save CSVs
    train_path = os.path.join(config.PROC_DATA_DIR, "train.csv")
    test_path  = os.path.join(config.PROC_DATA_DIR, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,   index=False)
    log(f"  Saved train.csv  ({os.path.getsize(train_path) / 1024:.1f} KB)")
    log(f"  Saved test.csv   ({os.path.getsize(test_path)  / 1024:.1f} KB)")

    # Save ID maps (keys stored as strings for JSON compatibility)
    user_map_path = os.path.join(config.PROC_DATA_DIR, "user_map.json")
    item_map_path = os.path.join(config.PROC_DATA_DIR, "item_map.json")
    with open(user_map_path, "w") as f:
        json.dump({str(k): v for k, v in user_map.items()}, f)
    with open(item_map_path, "w") as f:
        json.dump({str(k): v for k, v in item_map.items()}, f)
    log(f"  Saved user_map.json ({len(user_map)} users)")
    log(f"  Saved item_map.json ({len(item_map)} items)")

    # Save stats
    stats = {
        "n_users":             int(train_df["user_id"].nunique()),
        "n_items":             int(train_df["item_id"].nunique()),
        "n_train_interactions": int(len(train_df)),
        "n_test_interactions":  int(len(test_df)),
        "n_total_interactions": int(len(train_df) + len(test_df)),
        "avg_train_per_user":   round(len(train_df) / train_df["user_id"].nunique(), 2),
        "avg_test_per_user":    round(len(test_df)  / test_df["user_id"].nunique(), 2),
        "seed":                config.SEED,
        "min_user_ratings":    config.MIN_USER_RATINGS,
        "min_item_ratings":    config.MIN_ITEM_RATINGS,
    }
    stats_path = os.path.join(config.PROC_DATA_DIR, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    log(f"  Saved stats.json")

    return stats


def print_summary(stats):
    log("")
    log("=" * 55)
    log("PREPROCESSING COMPLETE — Dataset Summary")
    log("=" * 55)
    log(f"  Users:                  {stats['n_users']}")
    log(f"  Items:                  {stats['n_items']}")
    log(f"  Train interactions:     {stats['n_train_interactions']:,}")
    log(f"  Test  interactions:     {stats['n_test_interactions']:,}")
    log(f"  Avg train per user:     {stats['avg_train_per_user']}")
    log(f"  Avg test  per user:     {stats['avg_test_per_user']}")
    log("=" * 55)
    log("Next step: python 01_data/graph_builder.py")


def main():
    log("=" * 55)
    log("Preprocessing — MovieLens 100K")
    log("=" * 55)

    ratings_path = os.path.join(
        config.RAW_DATA_DIR, config.MOVIELENS_DIR, config.RATINGS_FILE
    )

    # Step 1: Load
    df = load_raw_ratings(ratings_path)

    # Step 2: Filter
    df = filter_by_frequency(df)

    # Step 3: Implicit feedback
    df = to_implicit(df)

    # Step 4: Remap IDs
    df, user_map, item_map = remap_ids(df)

    # Step 5: Train/test split
    train_df, test_df = train_test_split_per_user(df)

    # Step 6: Save
    stats = save_outputs(train_df, test_df, user_map, item_map)

    # Step 7: Summary
    print_summary(stats)


if __name__ == "__main__":
    main()

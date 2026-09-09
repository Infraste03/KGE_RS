# fashion_generalization/preprocessing/validate_inter.py

import pandas as pd
import random
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FASHION_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

INTER_FILE = os.path.join(
    FASHION_ROOT,
    "data",
    "recbole",
    "fashion",
    "fashion.inter"
)

SOURCE_FILE = os.path.join(
    FASHION_ROOT,
    "data",
    "interactions_5_core_filtered.csv"
)

random.seed(42)

# ── Load the two files ────────────────────────────────────────────────────────
inter = pd.read_csv(INTER_FILE, sep='\t')
source = pd.read_csv(SOURCE_FILE)

print("=== .inter FILE STRUCTURE ===")
print(inter.head(10).to_string())

print("\n=== SOURCE FILE STRUCTURE ===")
print(source.head(10).to_string())

# ── Check 1: same users ──────────────────────────────────────────────────────
users_inter = set(inter["user_id:token"])
users_source = set(source["reviewerID"])

print("\n=== USER CHECK ===")
print(f"  Users in .inter: {len(users_inter):,}")
print(f"  Users in source: {len(users_source):,}")
print(f"  Equal: {users_inter == users_source}")

# ── Check 2: same items ──────────────────────────────────────────────────────
items_inter = set(inter["item_id:token"])
items_source = set(source["asin"])

print("\n=== ITEM CHECK ===")
print(f"  Items in .inter: {len(items_inter):,}")
print(f"  Items in source: {len(items_source):,}")
print(f"  .inter is a subset of source: {items_inter <= items_source}")

# ── Check 3: spot check on 5 random users ────────────────────────────────────
print("\n=== SPOT CHECK ON 5 RANDOM USERS ===")

sample_users = random.sample(list(users_inter), 5)


def normalize_sequence(df, user_col, item_col, ts_col):
    return (
        df.sort_values([ts_col, item_col])[[item_col, ts_col]]
        .drop_duplicates()
        .values
        .tolist()
    )


for user in sample_users:
    seq_inter = normalize_sequence(
        inter[inter["user_id:token"] == user],
        "user_id:token",
        "item_id:token",
        "timestamp:float",
    )

    seq_source = normalize_sequence(
        source[source["reviewerID"] == user],
        "reviewerID",
        "asin",
        "unixReviewTime",
    )

    items_i = [item for item, _ in seq_inter]
    items_s = [item for item, _ in seq_source]

    match = items_i == items_s

    print(
        f"  User {user[:15]}... → "
        f"{len(items_i)} items → match: {match}"
    )

    if not match:
        print(f"    .inter: {items_i[:5]}")
        print(f"    source: {items_s[:5]}")


# ── Check 4: validation against the original raw file ────────────────────────
import gzip
import json

RAW_REVIEW_FILE = os.path.join(
    FASHION_ROOT,
    "data",
    "raw",
    "Clothing_Shoes_and_Jewelry.json.gz"
)

print("\n=== SPOT CHECK AGAINST RAW FILE ===")
print("  Reading raw file (this may take a few minutes)...")

# Use users from the previous random sample
check_users = set(sample_users[:200])

raw_records = {}  # user -> list of (timestamp, item)

with gzip.open(RAW_REVIEW_FILE, "rt", encoding="utf-8") as f:
    for line in f:
        try:
            r = json.loads(line.strip())
        except:
            continue

        user = r.get("reviewerID")
        item = r.get("asin")
        ts = r.get("unixReviewTime", 0)

        if user in check_users:
            if user not in raw_records:
                raw_records[user] = []

            raw_records[user].append((ts, item))

print(f"  Found {len(raw_records)} users in raw data.")

for user in check_users:
    if user not in raw_records:
        print(f"  User {user[:15]}... → NOT FOUND in raw data")
        continue

    # Sequence in the .inter file
    seq_inter = normalize_sequence(
        inter[inter["user_id:token"] == user],
        "user_id:token",
        "item_id:token",
        "timestamp:float",
    )

    # Sequence in raw data:
    # sort by timestamp and remove duplicates while preserving order
    raw_sorted = sorted(
        raw_records[user],
        key=lambda x: (x[0], x[1])
    )

    seen = set()
    raw_dedup = []

    for ts, item in raw_sorted:
        if (ts, item) not in seen:
            seen.add((ts, item))
            raw_dedup.append([item, float(ts)])

    # Compare only items that survive the filtering
    valid_items_in_inter = set(inter["item_id:token"])

    raw_filtered = [
        r
        for r in raw_dedup
        if r[0] in valid_items_in_inter
    ]

    match = seq_inter == raw_filtered

    print(
        f"  User {user[:15]}... → "
        f"match with raw data: {match}"
    )

    if not match:
        print(f"    .inter ({len(seq_inter)}): {seq_inter[:200]}")
        print(
            f"    raw filtered ({len(raw_filtered)}): "
            f"{raw_filtered[:200]}"
        )
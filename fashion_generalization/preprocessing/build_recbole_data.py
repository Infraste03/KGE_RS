# fashion_generalization/preprocessing/build_recbole_data.py

"""
Prepares RecBole data from the filtered Fashion interactions.

Output:
  - fashion.inter in RecBole format

The leave-one-out split is handled by RecBole.
"""

import pandas as pd
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FASHION_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

INPUT_FILE = os.path.join(
    FASHION_ROOT,
    "data",
    "interactions_5_core_filtered.csv"
)

OUT_DIR = os.path.join(
    FASHION_ROOT,
    "data",
    "recbole",
    "fashion"
)

os.makedirs(OUT_DIR, exist_ok=True)

# ── STEP 1: read filtered interactions ────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Reading filtered interactions...")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"  Rows read: {len(df):,}")
print(f"  Columns:   {list(df.columns)}")

# ── STEP 2: rename columns for RecBole ────────────────────────────────────────
print()
print("=" * 60)
print("STEP 2 — Renaming columns...")
print("=" * 60)

df = df.rename(columns={
    "reviewerID":      "user_id:token",
    "asin":            "item_id:token",
    "unixReviewTime":  "timestamp:float"
})

print(f"  Renamed columns: {list(df.columns)}")

# ── STEP 3: remove duplicates (same user, same item) ─────────────────────────
print()
print("=" * 60)
print("STEP 3 — Removing duplicates...")
print("=" * 60)

n_before = len(df)

df = df.drop_duplicates(
    subset=["user_id:token", "item_id:token"]
)

n_after = len(df)

print(f"  Rows before:       {n_before:,}")
print(f"  Rows after:        {n_after:,}")
print(f"  Duplicates removed: {n_before - n_after:,}")

# ── STEP 4: sort by user and timestamp ───────────────────────────────────────
print()
print("=" * 60)
print("STEP 4 — Sorting by user and timestamp...")
print("=" * 60)

df = df.sort_values([
    "user_id:token",
    "timestamp:float",
    "item_id:token"
])

print("  Sorting completed.")

# ── STEP 5: final statistics ─────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 5 — Final statistics...")
print("=" * 60)

n_users = df["user_id:token"].nunique()
n_items = df["item_id:token"].nunique()
n_inter = len(df)

seq_lengths = df.groupby("user_id:token").size()

print(f"  Users:                    {n_users:,}")
print(f"  Items:                    {n_items:,}")
print(f"  Interactions:             {n_inter:,}")
print(f"  Average sequence length:  {seq_lengths.mean():.1f}")
print(f"  Minimum sequence length:  {seq_lengths.min()}")
print(f"  Maximum sequence length:  {seq_lengths.max()}")

# Check that every user has at least 3 interactions
# (train + validation + test for leave-one-out evaluation)
n_short = (seq_lengths < 3).sum()

if n_short > 0:
    print(f"  WARNING: {n_short} users have fewer than 3 interactions")
    print("  (they will be automatically excluded by RecBole in the LOO split)")
else:
    print("  OK: all users have at least 3 interactions")

# ── STEP 6: write .inter file ────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 6 — Writing fashion.inter...")
print("=" * 60)

out_path = os.path.join(
    OUT_DIR,
    "fashion.inter"
)

df.to_csv(
    out_path,
    sep='\t',
    index=False
)

print(f"  Written: {out_path}")

print()
print("=" * 60)
print("RECBOLE DATA BUILD COMPLETED")
print("=" * 60)

print()
print("Next step: configure RecBole using these settings:")
print("  dataset: fashion")
print("  data_path: data/recbole")
print("  eval_args:")
print("    split: {LS: [0.8, 0.1, 0.1]}")
print("    order: TO")
print("    mode: full")
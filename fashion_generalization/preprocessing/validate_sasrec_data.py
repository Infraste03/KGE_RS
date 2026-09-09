# fashion_generalization/preprocessing/validate_sasrec_data.py

import os
import pandas as pd
from collections import Counter

# --- Paths --------------------------------────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FASHION_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))

INTER_FILE = os.path.join(
    FASHION_ROOT,
    "data",
    "recbole",
    "fashion",
    "fashion.inter"
)

print("=" * 60)
print("SASREC DATA VALIDATION")
print("=" * 60)

df = pd.read_csv(INTER_FILE, sep='\t')

# --- Check 1: column structure -----------------------------
print("\n[1] Column structure:")
print(f"  Columns: {list(df.columns)}")

expected = {
    "user_id:token",
    "item_id:token",
    "timestamp:float"
}

assert set(df.columns) == expected, "ERROR: incorrect columns"

print("  OK")

# --- Check 2: no null values -----------------------------------------
print("\n[2] Null values:")

nulls = df.isnull().sum()

if nulls.sum() == 0:
    print("  OK: no null values")
else:
    print(f"  WARNING: {nulls}")

# --- Check 3: sequence-length distribution -----------------------------------------
print("\n[3] Sequence-length distribution:")

seq_len = df.groupby("user_id:token").size()

print(f"  Total users:       {len(seq_len):,}")
print(f"  Mean:              {seq_len.mean():.1f}")
print(f"  Median:            {seq_len.median():.0f}")
print(f"  Min:               {seq_len.min()}")
print(f"  Max:               {seq_len.max()}")
print(f"  Users with seq<3:  {(seq_len < 3).sum()} (excluded by RecBole LOO)")
print(f"  Users with seq>=3: {(seq_len >= 3).sum()} (usable)")

# --- Check 4: item popularity distribution -----------------------------------------
print("\n[4] Item popularity:")

item_pop = df["item_id:token"].value_counts()

print(f"  Total items:                    {len(item_pop):,}")
print(f"  Average interactions per item:  {item_pop.mean():.1f}")
print(f"  Items with 1 interaction:       {(item_pop == 1).sum():,}")
print(f"  Items with >=5 interactions:    {(item_pop >= 5).sum():,}")
print("  Top 5 most popular items:")

for item, count in item_pop.head(5).items():
    print(f"    {item}: {count} interactions")

# --- Check 5: timestamp range -----------------------------------------
print("\n[5] Timestamp range:")

ts_min = pd.to_datetime(
    df["timestamp:float"].min(),
    unit='s'
)

ts_max = pd.to_datetime(
    df["timestamp:float"].max(),
    unit='s'
)

print(f"  From: {ts_min.date()}")
print(f"  To:   {ts_max.date()}")

if ts_min.year < 1990 or ts_max.year > 2025:
    print("  WARNING: timestamps outside expected range")
else:
    print("  OK: reasonable temporal range")

# ── Check 6: sparsity --------------------------------
print("\n[6] Sparsity:")

n_users = df["user_id:token"].nunique()
n_items = df["item_id:token"].nunique()
n_inter = len(df)

sparsity = 1 - (n_inter / (n_users * n_items))

print(f"  Sparsity: {sparsity:.6f} ({sparsity*100:.4f}%)")
print("  (typical for real-world datasets: >99%)")

print("\n" + "=" * 60)
print("SASREC VALIDATION COMPLETED")
print("=" * 60)
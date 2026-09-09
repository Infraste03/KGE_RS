"""
convert_v3_to_recbole.py — Converts Fashion v3 interactions
(selected favorable users) into the RecBole atomic .inter format,
ready for SASRec HPO.

Input:
  data_v3/interactions_5_core_filtered.csv

Expected columns:
  reviewerID, asin, unixReviewTime

Output:
  data/recbole_v3/fashion_v3/fashion_v3.inter
"""

import pandas as pd
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FASHION_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

INPUT_CSV = os.path.join(
    FASHION_ROOT,
    "data_v3",
    "interactions_5_core_filtered.csv"
)

OUT_DIR = os.path.join(
    FASHION_ROOT,
    "data",
    "recbole_v3",
    "fashion_v3"
)

OUT_FILE = os.path.join(
    OUT_DIR,
    "fashion_v3.inter"
)

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("Fashion v3 -> RecBole (.inter) conversion")
print("=" * 60)

df = pd.read_csv(INPUT_CSV)

print(f"  Interactions read: {len(df):,}")
print(f"  Unique users:      {df['reviewerID'].nunique():,}")
print(f"  Unique items:      {df['asin'].nunique():,}")

# ── Rename columns to the format expected by RecBole ─────────────────────────
df_out = df.rename(columns={
    "reviewerID":     "user_id:token",
    "asin":           "item_id:token",
    "unixReviewTime": "timestamp:float",
})[["user_id:token", "item_id:token", "timestamp:float"]]

# ── Sort by user and timestamp for sequential recommendation ─────────────────
df_out = df_out.sort_values([
    "user_id:token",
    "timestamp:float"
])

# ── Write output ──────────────────────────────────────────────────────────────
df_out.to_csv(
    OUT_FILE,
    sep="\t",
    index=False
)

print(f"  Written: {OUT_FILE}")
print(f"  Total rows: {len(df_out):,}")

print("=" * 60)
print("CONVERSION COMPLETED")
print("=" * 60)
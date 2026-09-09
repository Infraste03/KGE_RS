"""
Checks the coverage of the price and rank fields in the metadata
to determine whether they are sufficiently populated to be added to the KG.
"""

import os
import gzip
import json

# ── Paths ─────────────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FASHION_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))

META_FILE = os.path.join(
    FASHION_ROOT,
    "data",
    "raw",
    "meta_Clothing_Shoes_and_Jewelry.json.gz"
)

total = 0
with_price = 0
with_rank = 0
price_samples = []
rank_samples = []

with gzip.open(META_FILE, "rt", encoding="utf-8") as f:
    for line in f:
        try:
            r = json.loads(line.strip())
        except:
            continue

        total += 1

        price = r.get("price")
        if price:
            with_price += 1
            if len(price_samples) < 5:
                price_samples.append(price)

        rank = r.get("rank")
        if rank:
            with_rank += 1
            if len(rank_samples) < 5:
                rank_samples.append(rank)

        if total % 500000 == 0:
            print(f"  ...{total:,} records read")

print(f"\nTotal records:     {total:,}")
print(f"With price:        {with_price:,} ({with_price/total*100:.1f}%)")
print(f"With rank:         {with_rank:,} ({with_rank/total*100:.1f}%)")
print(f"\nPrice examples: {price_samples}")
print(f"Rank examples:  {rank_samples}")
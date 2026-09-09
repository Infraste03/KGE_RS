# fashion_generalization/exploration/inspect_meta_2018.py

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
has_also_buy = 0
has_categories = 0
sample_with_also_buy = None

with gzip.open(META_FILE, "rt", encoding="utf-8") as f:
    for line in f:
        try:
            record = json.loads(line.strip())
        except:
            continue

        total += 1

        if record.get("also_buy"):
            has_also_buy += 1

            if sample_with_also_buy is None:
                sample_with_also_buy = record

        if record.get("category"):
            has_categories += 1

print(f"Total records:             {total:,}")
print(f"With populated also_buy:   {has_also_buy:,}")
print(f"With populated category:   {has_categories:,}")

if sample_with_also_buy:
    print("\n=== EXAMPLE RECORD WITH also_buy ===")

    for key, value in sample_with_also_buy.items():
        print(f"  {key}: {repr(value)[:300]}")
else:
    print("\nNo record with also_buy found.")
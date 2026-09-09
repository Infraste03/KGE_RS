# fashion_generalization/exploration/inspect_meta.py

import os
import gzip
import json

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FASHION_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))

META_FILE = os.path.join(
    FASHION_ROOT,
    "data",
    "raw",
    "meta_Amazon_Fashion.jsonl.gz"
)

total = 0
with_categories = 0
with_bought_together = 0

with gzip.open(META_FILE, "rt", encoding="utf-8") as f:
    for line in f:
        try:
            record = json.loads(line.strip())
        except:
            continue

        total += 1

        if record.get("categories"):
            with_categories += 1

        if record.get("bought_together"):
            with_bought_together += 1

print(f"Total metadata records:        {total:,}")
print(f"With populated categories:    {with_categories:,}")
print(f"With populated bought_together: {with_bought_together:,}")
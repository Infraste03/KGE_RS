"""
fix_entity2id_v3.py — Rebuilds entity2id.tsv for v3 using only the entities
actually present in kg_train.tsv, removing orphan entities
(e.g., brands of products never purchased by the selected 50K users).

Does not modify kg_train.tsv, taskA_valid.tsv, or taskA_test.tsv
(which are already correct).

Overwrites data/processed_v3/entity2id.tsv with the cleaned version.
Saves a backup copy of the previous file before overwriting it.
"""

import pandas as pd
import os
import shutil

# ── Paths ─────────────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FASHION_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
PROCESSED_V3_DIR = os.path.join(FASHION_ROOT, "data", "processed_v3")

KG_TRAIN_PATH = os.path.join(PROCESSED_V3_DIR, "kg_train.tsv")
ENTITY2ID_PATH = os.path.join(PROCESSED_V3_DIR, "entity2id.tsv")
BACKUP_PATH = os.path.join(
    PROCESSED_V3_DIR,
    "entity2id_OLD_with_orphans.tsv"
)

print("=" * 60)
print("FIX entity2id.tsv (v3) — removing orphan entities")
print("=" * 60)

# ── Backup of the previous file ───────────────────────────────────────────────
if os.path.exists(ENTITY2ID_PATH):
    shutil.copy(ENTITY2ID_PATH, BACKUP_PATH)
    print(f"  Backup saved: {BACKUP_PATH}")

# ── Entities actually used in kg_train.tsv ───────────────────────────────────
kg_df = pd.read_csv(KG_TRAIN_PATH, sep='\t')
used_entities = sorted(set(kg_df["head"]) | set(kg_df["tail"]))

old_entity_count = (
    len(pd.read_csv(ENTITY2ID_PATH, sep="\t"))
    if os.path.exists(ENTITY2ID_PATH)
    else "N/A"
)

print(f"  Entities in previous entity2id.tsv: {old_entity_count}")

# ── New mapping with sequential IDs over alphabetically sorted entities ──────
new_entity2id = {e: i for i, e in enumerate(used_entities)}

with open(ENTITY2ID_PATH, "w", encoding="utf-8") as f:
    f.write("entity\tid\n")
    for e, i in new_entity2id.items():
        f.write(f"{e}\t{i}\n")

print(f"  New entity2id.tsv written: {len(new_entity2id):,} entities")

# ── Distribution check by entity type (prefix) ───────────────────────────────
from collections import defaultdict

by_prefix = defaultdict(int)

for e in used_entities:
    by_prefix[e.split("_")[0]] += 1

print("  Entity distribution by type:")

for prefix, count in sorted(by_prefix.items()):
    print(f"    {prefix}: {count:,}")

print()
print("=" * 60)
print("FIX COMPLETED")
print("=" * 60)
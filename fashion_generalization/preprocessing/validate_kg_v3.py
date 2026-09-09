"""
validate_kg_v3.py — Independent validation script for the Fashion v3 dataset.

Run AFTER generating v1, v2, and v3 (build_kg_fashion.py,
build_kg_fashion_v2.py, build_kg_fashion_v3.py).

Does not modify any generated files; it only reads them.

Checks performed:
  1. User overlap across v1/v2/v3 (Jaccard + intersections)
  2. Duplicate triples in kg_train.tsv (v3)
  3. Leakage across Task A splits (train/valid/test) for compatible_with (v3)
  4. Orphan entities in entity2id.tsv with respect to kg_train.tsv (v3)
  5. Item consistency between interactions_5_core_filtered.csv and kg_train.tsv (v3)
  6. Comparison of the n_categories distribution between the full population
     and the v3 sample
"""

import pandas as pd
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FASHION_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))

PATH_V1_INTER = os.path.join(
    FASHION_ROOT,
    "data",
    "interactions_5_core_filtered.csv"
)

PATH_V2_INTER = os.path.join(
    FASHION_ROOT,
    "data_v2",
    "interactions_5_core_filtered.csv"
)

PATH_V3_INTER = os.path.join(
    FASHION_ROOT,
    "data_v3",
    "interactions_5_core_filtered.csv"
)

PATH_V3_KG_TRAIN = os.path.join(
    FASHION_ROOT,
    "data",
    "processed_v3",
    "kg_train.tsv"
)

PATH_V3_VALID = os.path.join(
    FASHION_ROOT,
    "data",
    "processed_v3",
    "taskA_valid.tsv"
)

PATH_V3_TEST = os.path.join(
    FASHION_ROOT,
    "data",
    "processed_v3",
    "taskA_test.tsv"
)

PATH_V3_ENTITY2ID = os.path.join(
    FASHION_ROOT,
    "data",
    "processed_v3",
    "entity2id.tsv"
)

PATH_V3_METADATA = os.path.join(
    FASHION_ROOT,
    "data_v3",
    "metadata_valid_items.json"
)

print("=" * 70)
print("FASHION KG v3 VALIDATION")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 1 — User overlap across v1, v2, v3
# ══════════════════════════════════════════════════════════════════════════════
print()
print("─" * 70)
print("CHECK 1 — User overlap across v1/v2/v3")
print("─" * 70)


def load_users(path):
    if not os.path.exists(path):
        print(f"  WARNING: file not found -> {path}")
        return None
    return set(pd.read_csv(path)["reviewerID"].unique())


users_v1 = load_users(PATH_V1_INTER)
users_v2 = load_users(PATH_V2_INTER)
users_v3 = load_users(PATH_V3_INTER)


def compare_users(name_a, set_a, name_b, set_b):
    if set_a is None or set_b is None:
        print(f"  Skipping comparison {name_a} vs {name_b}: missing file.")
        return

    inter = set_a & set_b
    union = set_a | set_b
    jaccard = len(inter) / len(union) if union else 0.0

    print(f"{name_a} ({len(set_a):,}) vs {name_b} ({len(set_b):,}):")
    print(f"  users in common:            {len(inter):,}")
    print(f"  overlap relative to {name_a}: {len(inter)/len(set_a)*100:.1f}%")
    print(f"  Jaccard:                    {jaccard:.4f}")
    print()


compare_users("v1", users_v1, "v2", users_v2)
compare_users("v1", users_v1, "v3", users_v3)
compare_users("v2", users_v2, "v3", users_v3)

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 2 — Duplicates in kg_train.tsv (v3)
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("CHECK 2 — Duplicates in kg_train.tsv (v3)")
print("─" * 70)

if os.path.exists(PATH_V3_KG_TRAIN):
    kg_df = pd.read_csv(PATH_V3_KG_TRAIN, sep='\t')
    dups = kg_df[kg_df.duplicated()]

    print(f"  Total triples:      {len(kg_df):,}")
    print(f"  Duplicate triples:  {len(dups):,}")

    if len(dups) > 0:
        print("  WARNING: duplicate triples found, first 5:")
        print(dups.head())
    else:
        print("  OK — no duplicates.")
else:
    print(f"  WARNING: file not found -> {PATH_V3_KG_TRAIN}")
    kg_df = None

print()

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 3 — Leakage across Task A splits (train/valid/test)
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("CHECK 3 — Leakage across Task A splits (compatible_with)")
print("─" * 70)

if kg_df is not None and os.path.exists(PATH_V3_VALID) and os.path.exists(PATH_V3_TEST):
    cw_train_df = kg_df[kg_df["relation"] == "compatible_with"]
    cw_valid_df = pd.read_csv(PATH_V3_VALID, sep='\t')
    cw_test_df  = pd.read_csv(PATH_V3_TEST, sep='\t')

    train_pairs = set(zip(cw_train_df["head"], cw_train_df["tail"]))
    valid_pairs = set(zip(cw_valid_df["head"], cw_valid_df["tail"]))
    test_pairs  = set(zip(cw_test_df["head"], cw_test_df["tail"]))

    leak_train_valid = train_pairs & valid_pairs
    leak_train_test  = train_pairs & test_pairs
    leak_valid_test  = valid_pairs & test_pairs

    print(f"  compatible_with train triples: {len(train_pairs):,}")
    print(f"  compatible_with valid triples: {len(valid_pairs):,}")
    print(f"  compatible_with test triples:  {len(test_pairs):,}")
    print(f"  Train/valid overlap: {len(leak_train_valid):,}")
    print(f"  Train/test overlap:  {len(leak_train_test):,}")
    print(f"  Valid/test overlap:  {len(leak_valid_test):,}")

    if leak_train_valid or leak_train_test or leak_valid_test:
        print("  WARNING: leakage detected across splits!")
    else:
        print("  OK — no leakage across splits.")
else:
    print("  WARNING: missing files, skipping check.")

print()

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 4 — Orphan entities in entity2id.tsv relative to kg_train.tsv
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("CHECK 4 — Orphan entities (in entity2id.tsv but never in kg_train.tsv)")
print("─" * 70)

if kg_df is not None and os.path.exists(PATH_V3_ENTITY2ID):
    entity2id_df = pd.read_csv(PATH_V3_ENTITY2ID, sep='\t')
    all_entities = set(entity2id_df["entity"])
    used_entities = set(kg_df["head"]) | set(kg_df["tail"])

    orphans = all_entities - used_entities
    orphans_by_prefix = {}

    for e in orphans:
        prefix = e.split("_")[0]
        orphans_by_prefix[prefix] = orphans_by_prefix.get(prefix, 0) + 1

    print(f"  Total entities in entity2id.tsv: {len(all_entities):,}")
    print(f"  Entities used in kg_train.tsv:   {len(used_entities):,}")
    print(f"  Orphan entities:                 {len(orphans):,}")
    print(f"  Orphan distribution by type:     {orphans_by_prefix}")
    print("  NOTE: orphan 'pricetier_'/'poptier_' entities are expected if")
    print("        those relations were filtered post-hoc. Orphan")
    print("        'item_'/'category_'/'brand_' entities should never appear;")
    print("        investigate if any are found.")
else:
    print("  WARNING: missing files, skipping check.")

print()

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 5 — Item consistency between Task B (interactions) and Task A (KG)
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("CHECK 5 — Item consistency between interactions (Task B) and kg_train (Task A)")
print("─" * 70)

if kg_df is not None and os.path.exists(PATH_V3_INTER):
    inter_df = pd.read_csv(PATH_V3_INTER)
    taskb_items = set(inter_df["asin"].unique())
    taskb_items_prefixed = {f"item_{i}" for i in taskb_items}

    kg_items = {e for e in used_entities if e.startswith("item_")}

    only_in_taskb = taskb_items_prefixed - kg_items
    only_in_kg    = kg_items - taskb_items_prefixed

    print(f"  Items in Task B (interactions): {len(taskb_items_prefixed):,}")
    print(f"  Items in Task A (KG):           {len(kg_items):,}")
    print(f"  Items only in Task B (never in KG): {len(only_in_taskb):,}")
    print(f"  Items only in KG (never in interactions): {len(only_in_kg):,}")

    if only_in_taskb:
        print("  NOTE: items only in Task B are expected if they never appear")
        print("        in compatible_with/belongs_to triples (isolated KG items).")
else:
    print("  WARNING: missing files, skipping check.")

print()

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 6 — n_categories distribution: full population vs v3 sample
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("CHECK 6 — n_categories distribution (population vs v3 sample)")
print("─" * 70)

import json

if os.path.exists(PATH_V3_METADATA) and os.path.exists(PATH_V3_INTER):
    with open(PATH_V3_METADATA) as f:
        metadata = json.load(f)

    item_category = {
        item: meta["categories"]
        for item, meta in metadata.items()
    }

    inter_df = pd.read_csv(PATH_V3_INTER)
    inter_df["categoria"] = inter_df["asin"].map(item_category)

    n_cat_per_user = inter_df.groupby("reviewerID")["categoria"].nunique()
    distribuzione = n_cat_per_user.value_counts().sort_index()

    print("  n_categories distribution in the selected v3 sample:")

    for n_cat, count in distribuzione.items():
        pct = count / len(n_cat_per_user) * 100
        print(f"    {n_cat} categories: {count:,} users ({pct:.1f}%)")

    print()
    print("  NOTE: compare these values with those printed during Step 5")
    print("        of build_kg_fashion_v3.py (distribution BEFORE selection,")
    print("        i.e., over the entire post-5-core population) to quantify")
    print("        how much the selection changed the distribution.")
else:
    print("  WARNING: missing files, skipping check.")

print()
print("=" * 70)
print("VALIDATION COMPLETED")
print("=" * 70)
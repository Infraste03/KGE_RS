# fashion_generalization/preprocessing/validate_transe_data.py

import os
import pandas as pd
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FASHION_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
PROCESSED_DIR = os.path.join(FASHION_ROOT, "data", "processed")

KG_TRAIN = os.path.join(PROCESSED_DIR, "kg_train.tsv")
KG_VALID = os.path.join(PROCESSED_DIR, "taskA_valid.tsv")
KG_TEST  = os.path.join(PROCESSED_DIR, "taskA_test.tsv")
E2ID     = os.path.join(PROCESSED_DIR, "entity2id.tsv")
R2ID     = os.path.join(PROCESSED_DIR, "relation2id.tsv")

print("=" * 60)
print("TRANSE DATA VALIDATION")
print("=" * 60)

kg    = pd.read_csv(KG_TRAIN, sep='\t')
valid = pd.read_csv(KG_VALID, sep='\t')
test  = pd.read_csv(KG_TEST, sep='\t')
e2id  = pd.read_csv(E2ID, sep='\t')
r2id  = pd.read_csv(R2ID, sep='\t')

# ── Check 1: structure ────────────────────────────────────────────────────────
print("\n[1] File sizes:")
print(f"  kg_train:     {len(kg):,} triples")
print(f"  taskA_valid:  {len(valid):,} triples")
print(f"  taskA_test:   {len(test):,} triples")
print(f"  entity2id:    {len(e2id):,} entities")
print(f"  relation2id:  {len(r2id):,} relations")

# ── Check 2: no null values ───────────────────────────────────────────────────
print("\n[2] Null values in KG train:")
nulls = kg.isnull().sum()
if nulls.sum() == 0:
    print("  OK: no null values")
else:
    print(f"  ERROR: {nulls}")

# ── Check 3: all entities are included in the mapping ────────────────────────
print("\n[3] entity2id coverage:")
all_entities_in_kg = set(kg['head']) | set(kg['tail'])
all_entities_in_map = set(e2id['entity'])
missing = all_entities_in_kg - all_entities_in_map

if len(missing) == 0:
    print(f"  OK: all {len(all_entities_in_kg):,} entities are included in the mapping")
else:
    print(f"  ERROR: {len(missing):,} entities missing from the mapping")
    print(f"  Example: {list(missing)[:3]}")

# ── Check 4: relation distribution ────────────────────────────────────────────
print("\n[4] Relation distribution in KG train:")
for rel, count in kg['relation'].value_counts().items():
    print(f"  {rel:<25} {count:>8,} triples")

# ── Check 5: node degree for compatible_with ─────────────────────────────────
print("\n[5] Node degree — compatible_with:")
cw = kg[kg['relation'] == 'compatible_with']
degree = cw.groupby('head').size()

print(f"  Items with at least 1 pair: {len(degree):,}")
print(f"  Average degree:             {degree.mean():.2f}")
print(f"  Median degree:              {degree.median():.1f}")
print(f"  Maximum degree:             {degree.max()}")
print(f"  Items with degree 1:        {(degree == 1).sum():,}")
print(f"  Items with degree >= 3:     {(degree >= 3).sum():,}")

# ── Check 6: no validation/test leakage into train ────────────────────────────
print("\n[6] Leakage check:")
train_set = set(zip(kg['head'], kg['relation'], kg['tail']))

leaks_v = sum(
    1 for _, r in valid.iterrows()
    if (r['head'], r['relation'], r['tail']) in train_set
)

leaks_t = sum(
    1 for _, r in test.iterrows()
    if (r['head'], r['relation'], r['tail']) in train_set
)

if leaks_v == 0 and leaks_t == 0:
    print("  OK: no leakage")
else:
    print(
        f"  ERROR: {leaks_v} validation + "
        f"{leaks_t} test triples found in train"
    )

# ── Check 7: validation/test entities covered by train ────────────────────────
print("\n[7] Validation/test entities covered by train:")
train_entities = set(kg['head']) | set(kg['tail'])
valid_entities = set(valid['head']) | set(valid['tail'])
test_entities  = set(test['head']) | set(test['tail'])

v_covered = valid_entities & train_entities
t_covered = test_entities & train_entities

print(
    f"  Valid: {len(v_covered)}/{len(valid_entities)} entities in train "
    f"({len(v_covered)/len(valid_entities)*100:.1f}%)"
)

print(
    f"  Test:  {len(t_covered)}/{len(test_entities)} entities in train "
    f"({len(t_covered)/len(test_entities)*100:.1f}%)"
)

# ── Check 8: cross-category compatible_with pair balance ─────────────────────
print("\n[8] compatible_with pair balance by type:")

entity_cat = {}
bt = kg[kg['relation'] == 'belongs_to']

for _, row in bt.iterrows():
    item = row['head']
    cat  = row['tail'].replace('category_', '')
    entity_cat[item] = cat

pair_counts = defaultdict(int)

for _, row in cw.iterrows():
    cat_a = entity_cat.get(row['head'], 'unknown')
    cat_b = entity_cat.get(row['tail'], 'unknown')
    pair  = "-".join(sorted([cat_a, cat_b]))
    pair_counts[pair] += 1

for pair, count in sorted(pair_counts.items(), key=lambda x: -x[1]):
    print(f"  {pair:<25} {count:>6,} triples")

print("\n" + "=" * 60)
print("TRANSE VALIDATION COMPLETED")
print("=" * 60)
# fashion_generalization/preprocessing/build_kg_fashion.py

"""
Builds the Fashion Knowledge Graph from the
Clothing, Shoes and Jewelry 2018 dataset (McAuley).

Entities:
  - item_* : Fashion items
  - category_* : macro-categories (top, bottom, shoes)
  - brand_* : product brands

Relations:
  - compatible_with  : item -> item (from cross-category also_buy pairs)
  - belongs_to       : item -> category
  - belongs_to_brand : item -> brand

Outputs in data/processed/:
  - kg_train.tsv       (triples for TransE)
  - taskA_valid.tsv    (Task A validation holdout)
  - taskA_test.tsv     (Task A test holdout)
  - entity2id.tsv      (entity -> integer ID mapping)
  - relation2id.tsv    (relation -> integer ID mapping)
  - kg_stats.txt       (graph statistics)
"""

import gzip
import json
import random
import os
from collections import defaultdict
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FASHION_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))

REVIEW_FILE = os.path.join(
    FASHION_ROOT,
    "data",
    "raw",
    "Clothing_Shoes_and_Jewelry.json.gz"
)

META_FILE = os.path.join(
    FASHION_ROOT,
    "data",
    "raw",
    "meta_Clothing_Shoes_and_Jewelry.json.gz"
)

OUT_DIR = os.path.join(
    FASHION_ROOT,
    "data",
    "processed"
)
os.makedirs(OUT_DIR, exist_ok=True)

# ── Parameters — identical to the exploration setup ──────────────────────────
MIN_INTERACTIONS = 5
N_USERS          = 50_000
SEED             = 42
TASKA_TRAIN_RATIO = 0.90
TASKA_VALID_RATIO = 0.05
# TASKA_TEST_RATIO is the remaining fraction

KEEP_CATEGORIES = {"top", "bottom", "shoes"}

CATEGORY_MAP = {
    "top":    ["tops", "tees", "blouses", "shirts", "sweaters",
               "sweatshirts", "hoodies", "tank", "tunic", "polos"],
    "bottom": ["pants", "jeans", "shorts", "skirts", "leggings",
               "trousers", "chinos"],
    "shoes":  ["shoes", "sneakers", "boots", "sandals", "loafers",
               "heels", "flats", "oxfords", "slippers", "flip-flops"],
}

random.seed(SEED)


def get_macro_category(category_list):
    if not category_list:
        return None
    text = " ".join(category_list).lower()
    for macro, keywords in CATEGORY_MAP.items():
        for kw in keywords:
            if kw in text:
                return macro
    return None


# ── STEP 1: read reviews ──────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Reading reviews...")
print("=" * 60)

user_items = defaultdict(list)
item_users = defaultdict(set)
total = 0

with gzip.open(REVIEW_FILE, "rt", encoding="utf-8") as f:
    for line in f:
        try:
            r = json.loads(line.strip())
        except:
            continue

        user = r.get("reviewerID")
        item = r.get("asin")
        ts   = r.get("unixReviewTime", 0)

        if user and item:
            user_items[user].append((ts, item))
            item_users[item].add(user)
            total += 1

        if total % 1_000_000 == 0:
            print(f"  ...{total:,} reviews read")

print(f"  Total reviews: {total:,}")

# ── STEP 2: read metadata ─────────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 2 — Reading metadata...")
print("=" * 60)

item_category = {}
item_also_buy = {}
item_brand = {}  # Dictionary storing item brands
total_meta = 0

with gzip.open(META_FILE, "rt", encoding="utf-8") as f:
    for line in f:
        try:
            r = json.loads(line.strip())
        except:
            continue

        total_meta += 1
        item = r.get("asin")

        if not item:
            continue

        cats = r.get("category", [])
        macro = get_macro_category(cats)

        if macro:
            item_category[item] = macro

        # Extract brand
        brand = r.get("brand")
        if brand:
            item_brand[item] = brand

        ab = r.get("also_buy")
        if ab:
            item_also_buy[item] = ab

        if total_meta % 500_000 == 0:
            print(f"  ...{total_meta:,} metadata records read")

print(f"  Products in metadata: {total_meta:,}")

# ── STEP 3: category filtering ────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 3 — Filtering by Fashion category...")
print("=" * 60)

fashion_items = {i for i, c in item_category.items() if c in KEEP_CATEGORIES}

user_items_f = defaultdict(list)
item_users_f = defaultdict(set)

for user, interactions in user_items.items():
    filtered = [(ts, i) for ts, i in interactions if i in fashion_items]

    if filtered:
        user_items_f[user] = filtered

        for _, item in filtered:
            item_users_f[item].add(user)

# ── STEP 4: 5-core filtering ──────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 4 — Iterative 5-core filtering...")
print("=" * 60)

user_items = user_items_f
item_users = item_users_f

iteration = 0

while True:
    iteration += 1

    valid_items = {
        i for i, u in item_users.items()
        if len(u) >= MIN_INTERACTIONS
    }

    new_user_items = {}

    for user, interactions in user_items.items():
        filtered = [
            (ts, i)
            for ts, i in interactions
            if i in valid_items
        ]

        if len(filtered) >= MIN_INTERACTIONS:
            new_user_items[user] = filtered

    new_item_users = defaultdict(set)

    for user, interactions in new_user_items.items():
        for _, item in interactions:
            new_item_users[item].add(user)

    if (len(new_user_items) == len(user_items) and
            len(new_item_users) == len(item_users)):
        print(f"  Convergence reached at iteration {iteration}.")
        break

    user_items = new_user_items
    item_users = new_item_users

    print(
        f"  Iteration {iteration}: "
        f"{len(user_items):,} users, {len(item_users):,} items"
    )

# ── STEP 5: sample 50K users ──────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 5 — Sampling 50,000 users...")
print("=" * 60)

all_valid_users = list(user_items.keys())
n_sample        = min(N_USERS, len(all_valid_users))
sampled_users   = set(random.sample(all_valid_users, n_sample))

user_items = {u: user_items[u] for u in sampled_users}
item_users = defaultdict(set)

for user, interactions in user_items.items():
    for _, item in interactions:
        item_users[item].add(user)

valid_items_set = set(item_users.keys())

print(f"  Users: {len(user_items):,}")
print(f"  Items: {len(valid_items_set):,}")

# ── STEP 5.5: save intermediate validation data ───────────────────────────────
print()
print("=" * 60)
print("STEP 5.5 — Saving intermediate validation data...")
print("=" * 60)

# Output directory for intermediate validation data
VALIDATION_DATA_DIR = os.path.join(
    FASHION_ROOT,
    "data"
)
os.makedirs(VALIDATION_DATA_DIR, exist_ok=True)

# 1. Save filtered and sampled interactions
interactions_to_save = []

for user, inter in user_items.items():
    for ts, item in inter:
        interactions_to_save.append(
            {
                "reviewerID": user,
                "asin": item,
                "unixReviewTime": ts
            }
        )

df_interactions = pd.DataFrame(interactions_to_save)

interactions_path = os.path.join(
    VALIDATION_DATA_DIR,
    "interactions_5_core_filtered.csv"
)

df_interactions.to_csv(interactions_path, index=False)
print(f"  Written: {interactions_path}")

# 2. Save metadata only for valid items
metadata_to_save = {}

for item in valid_items_set:
    metadata_to_save[item] = {
        "categories": item_category.get(item),
        "brand": item_brand.get(item),
    }

metadata_path = os.path.join(
    VALIDATION_DATA_DIR,
    "metadata_valid_items.json"
)

with open(metadata_path, 'w') as f:
    json.dump(metadata_to_save, f)

print(f"  Written: {metadata_path}")

# 3. Save filtered also_buy pairs involving valid items only
also_buy_to_save = {}

for item_a, ab_list in item_also_buy.items():
    if item_a in valid_items_set:
        filtered_ab = [
            item_b
            for item_b in ab_list
            if item_b in valid_items_set
        ]

        if filtered_ab:
            also_buy_to_save[item_a] = filtered_ab

also_buy_path = os.path.join(
    VALIDATION_DATA_DIR,
    "also_buy_pairs.json"
)

with open(also_buy_path, 'w') as f:
    json.dump(also_buy_to_save, f)

print(f"  Written: {also_buy_path}")

# ── STEP 6: build KG triples ──────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 6 — Building KG triples...")
print("=" * 60)

compatible_with_triples = []   # (item_a, item_b)
belongs_to_triples = []        # (item, category)
belongs_to_brand_triples = []  # (item, brand)

# belongs_to: each valid item -> its category
for item in valid_items_set:
    cat = item_category.get(item)

    if cat:
        belongs_to_triples.append((item, cat))

for item in valid_items_set:
    brand = item_brand.get(item)

    if brand:
        # Remove tabs, newlines, and problematic characters
        brand_clean = (
            brand
            .replace('\t', ' ')
            .replace('\n', ' ')
            .replace('\r', '')
            .strip()
        )

        if brand_clean:
            belongs_to_brand_triples.append((item, brand_clean))

# compatible_with: cross-category also_buy pairs among valid items
# Use also_buy_to_save, which already contains valid item pairs only
seen_pairs = set()

for item_a, ab_list in also_buy_to_save.items():
    # item_a is already guaranteed to be in valid_items_set
    cat_a = item_category.get(item_a)

    for item_b in ab_list:
        # item_b is already guaranteed to be in valid_items_set
        cat_b = item_category.get(item_b)

        if cat_a is None or cat_b is None:
            continue

        if cat_a == cat_b:
            continue  # Discard same-category pairs

        pair = tuple(sorted([item_a, item_b]))

        if pair in seen_pairs:
            continue

        seen_pairs.add(pair)

        # Always store pairs in the same order
        compatible_with_triples.append(
            (min(item_a, item_b), max(item_a, item_b))
        )

print(f"  compatible_with triples:  {len(compatible_with_triples):,}")
print(f"  belongs_to triples:       {len(belongs_to_triples):,}")
print(f"  belongs_to_brand triples: {len(belongs_to_brand_triples):,}")

# ── STEP 7: Task A split (train / valid / test) ───────────────────────────────
print()
print("=" * 60)
print("STEP 7 — Task A split (90/5/5)...")
print("=" * 60)

random.shuffle(compatible_with_triples)

n     = len(compatible_with_triples)
n_tr  = int(n * TASKA_TRAIN_RATIO)
n_val = int(n * TASKA_VALID_RATIO)

cw_train = compatible_with_triples[:n_tr]
cw_valid = compatible_with_triples[n_tr:n_tr + n_val]
cw_test  = compatible_with_triples[n_tr + n_val:]

print(f"  Train: {len(cw_train):,}")
print(f"  Valid: {len(cw_valid):,}")
print(f"  Test:  {len(cw_test):,}")

# ── STEP 8: build entity2id and relation2id ───────────────────────────────────
print()
print("=" * 60)
print("STEP 8 — Building ID mapping...")
print("=" * 60)

# Collect all entities
all_entities = set()

for item in valid_items_set:
    all_entities.add(f"item_{item}")

for cat in KEEP_CATEGORIES:
    all_entities.add(f"category_{cat}")

all_brands = {
    b.replace('\t', ' ')
     .replace('\n', ' ')
     .replace('\r', '')
     .strip()
    for b in item_brand.values()
    if b
}

all_brands.discard('')

for brand in all_brands:
    all_entities.add(f"brand_{brand}")

entity2id = {
    e: i
    for i, e in enumerate(sorted(all_entities))
}

relation2id = {
    "compatible_with": 0,
    "belongs_to": 1,
    "belongs_to_brand": 2
}

print(f"  Total entities:  {len(entity2id):,}")
print(f"  Total relations: {len(relation2id):,}")

# ── STEP 9: write output files ────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 9 — Writing output files...")
print("=" * 60)


def write_triples(path, triples, relation, mode='w'):
    with open(path, mode, encoding="utf-8") as f:
        if mode == 'w':
            f.write("head\trelation\ttail\n")

        for head, tail in triples:
            f.write(f"item_{head}\t{relation}\titem_{tail}\n")


def write_belongs_to(path, triples, mode='w'):
    with open(path, mode, encoding="utf-8") as f:
        if mode == 'w':
            f.write("head\trelation\ttail\n")

        for item, cat in triples:
            f.write(f"item_{item}\tbelongs_to\tcategory_{cat}\n")


def write_belongs_to_brand(path, triples, mode='w'):
    with open(path, mode, encoding="utf-8") as f:
        if mode == 'w':
            f.write("head\trelation\ttail\n")

        for item, brand in triples:
            f.write(f"item_{item}\tbelongs_to_brand\tbrand_{brand}\n")


# kg_train.tsv: first call uses 'w', subsequent calls use 'a'
kg_train_path = os.path.join(OUT_DIR, "kg_train.tsv")

write_triples(
    kg_train_path,
    cw_train,
    "compatible_with",
    mode='w'
)

write_belongs_to(
    kg_train_path,
    belongs_to_triples,
    mode='a'
)

write_belongs_to_brand(
    kg_train_path,
    belongs_to_brand_triples,
    mode='a'
)

print("  Written: kg_train.tsv")

# taskA_valid.tsv
valid_path = os.path.join(OUT_DIR, "taskA_valid.tsv")

write_triples(
    valid_path,
    cw_valid,
    "compatible_with",
    mode='w'
)

print("  Written: taskA_valid.tsv")

# taskA_test.tsv
test_path = os.path.join(OUT_DIR, "taskA_test.tsv")

write_triples(
    test_path,
    cw_test,
    "compatible_with",
    mode='w'
)

print("  Written: taskA_test.tsv")

# entity2id.tsv
e2id_path = os.path.join(OUT_DIR, "entity2id.tsv")

with open(e2id_path, "w", encoding="utf-8") as f:
    f.write("entity\tid\n")

    for e, i in entity2id.items():
        f.write(f"{e}\t{i}\n")

print("  Written: entity2id.tsv")

# relation2id.tsv
r2id_path = os.path.join(OUT_DIR, "relation2id.tsv")

with open(r2id_path, "w", encoding="utf-8") as f:
    f.write("relation\tid\n")

    for r, i in relation2id.items():
        f.write(f"{r}\t{i}\n")

print("  Written: relation2id.tsv")


import pandas as pd

kg_df = pd.read_csv(kg_train_path, sep='\t')
dups = kg_df[kg_df.duplicated()]
print(dups)

# ── STEP 10: final statistics ─────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 10 — Final statistics...")
print("=" * 60)

total_kg_triples = (
    len(cw_train)
    + len(belongs_to_triples)
    + len(belongs_to_brand_triples)
)

stats = f"""
FASHION KG — FINAL STATISTICS
=============================
Sampled users:                  {len(user_items):,}
Valid items:                    {len(valid_items_set):,}
Total entities (KG):            {len(entity2id):,}
Relations:                      {len(relation2id):,}
Total compatible_with triples:  {len(compatible_with_triples):,}
  -> train:                     {len(cw_train):,}
  -> valid (Task A):            {len(cw_valid):,}
  -> test  (Task A):            {len(cw_test):,}

belongs_to triples:             {len(belongs_to_triples):,}
belongs_to_brand triples:       {len(belongs_to_brand_triples):,}
Total KG triples (train):       {total_kg_triples:,}

Split: {int(TASKA_TRAIN_RATIO*100)}/{int(TASKA_VALID_RATIO*100)}/{int((1-TASKA_TRAIN_RATIO-TASKA_VALID_RATIO)*100)}
Seed:  {SEED}
"""

print(stats)

stats_path = os.path.join(
    OUT_DIR,
    "kg_stats.txt"
)

with open(stats_path, "w", encoding="utf-8") as f:
    f.write(stats)

print("  Written: kg_stats.txt")

print()
print("=" * 60)
print("KG BUILD COMPLETED")
print("=" * 60)
# User selection favoring KG-rich profiles

"""
Fashion KG v3 — selects "favorable" users instead of random sampling.

Based on v2 (includes price_tier and popularity_tier).
Does NOT modify v1/v2 files in data/processed or data/processed_v2.
Outputs are written to data/processed_v3/.

Key difference compared with v1/v2:
  - Step 5 no longer uses random.sample() to select 50,000 users.
  - Users are ranked by:
      1) number of distinct purchased categories (top/bottom/shoes) — primary criterion
      2) total number of interactions — secondary criterion (tie-breaker)
    and the top N_USERS users are selected.
  - After selection, a 5-core convergence check is run again to ensure
    that the selected subset remains valid.

Relations (identical to v2):
  - compatible_with, belongs_to, belongs_to_brand,
    belongs_to_price_tier, belongs_to_pop_tier
"""

import gzip
import json
import random
import os
import re
from collections import defaultdict
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FASHION_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))

# Input paths — same source files used by v1/v2
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

# Output path — separate from v1/v2
OUT_DIR = os.path.join(
    FASHION_ROOT,
    "data",
    "processed_v3"
)
os.makedirs(OUT_DIR, exist_ok=True)

VALIDATION_DATA_DIR = os.path.join(
    FASHION_ROOT,
    "data_v3"
)
os.makedirs(VALIDATION_DATA_DIR, exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
MIN_INTERACTIONS  = 5
N_USERS           = 50_000
SEED              = 42
TASKA_TRAIN_RATIO = 0.90
TASKA_VALID_RATIO = 0.05

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


def parse_price(price_str):
    """'$31.99' -> 31.99. Returns None if the value cannot be parsed."""
    if not price_str or not isinstance(price_str, str):
        return None
    match = re.search(r'[\d,]+\.?\d*', price_str.replace('$', ''))
    if not match:
        return None
    try:
        return float(match.group().replace(',', ''))
    except ValueError:
        return None


def parse_rank(rank_str):
    """'19,963,069inClothing,ShoesJewelry(' -> 19963069. Returns None if not parsable."""
    if not rank_str or not isinstance(rank_str, str):
        return None
    match = re.match(r'^([\d,]+)', rank_str)
    if not match:
        return None
    try:
        return int(match.group().replace(',', ''))
    except ValueError:
        return None


# ── STEP 1: read reviews — identical to v1/v2 ────────────────────────────────
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

# ── STEP 2: read metadata (category, brand, price, rank) ─────────────────────
print()
print("=" * 60)
print("STEP 2 — Reading metadata...")
print("=" * 60)

item_category = {}
item_also_buy = {}
item_brand    = {}
item_price    = {}
item_rank     = {}
total_meta    = 0

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

        cats  = r.get("category", [])
        macro = get_macro_category(cats)
        if macro:
            item_category[item] = macro

        brand = r.get("brand")
        if brand:
            item_brand[item] = brand

        price = parse_price(r.get("price"))
        if price is not None:
            item_price[item] = price

        rank = parse_rank(r.get("rank"))
        if rank is not None:
            item_rank[item] = rank

        ab = r.get("also_buy")
        if ab:
            item_also_buy[item] = ab

        if total_meta % 500_000 == 0:
            print(f"  ...{total_meta:,} metadata records read")

print(f"  Products in metadata: {total_meta:,}")
print(f"  With price:           {len(item_price):,}")
print(f"  With rank:            {len(item_rank):,}")

# ── STEP 3: category filtering — identical to v1/v2 ─────────────────────────
print()
print("=" * 60)
print("STEP 3 — Filtering by Fashion category...")
print("=" * 60)

fashion_items = {i for i, c in item_category.items() if c in KEEP_CATEGORIES}

user_items_f = defaultdict(list)
item_users_f = defaultdict(set)
for user, interactions in user_items.items():
    filtered = [(ts, i) for ts, i in interactions
                if i in fashion_items and item_category.get(i) in KEEP_CATEGORIES]
    if filtered:
        user_items_f[user] = filtered
        for _, item in filtered:
            item_users_f[item].add(user)

# ── STEP 4: iterative 5-core filtering — identical to v1/v2 ─────────────────
print()
print("=" * 60)
print("STEP 4 — Iterative 5-core filtering...")
print("=" * 60)

user_items = user_items_f
item_users = item_users_f
iteration = 0

while True:
    iteration += 1
    valid_items = {i for i, u in item_users.items() if len(u) >= MIN_INTERACTIONS}
    new_user_items = {}
    for user, interactions in user_items.items():
        filtered = [(ts, i) for ts, i in interactions if i in valid_items]
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
    print(f"  Iteration {iteration}: {len(user_items):,} users, {len(item_users):,} items")

# ── STEP 5: select "favorable" users (NEW — replaces random.sample) ───────────
print()
print("=" * 60)
print(f"STEP 5 — Selecting the {N_USERS:,} most favorable users...")
print("=" * 60)

# For each user compute:
#   n_categorie   -> number of distinct purchased macro-categories (1, 2, or 3)
#   n_interazioni -> total number of interactions
user_scores = []
for user, interactions in user_items.items():
    n_interazioni = len(interactions)
    categorie_utente = {item_category.get(i) for _, i in interactions}
    categorie_utente.discard(None)
    n_categorie = len(categorie_utente)
    user_scores.append((user, n_categorie, n_interazioni))

# Sort first by decreasing n_categorie, then by decreasing n_interazioni
user_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)

n_sample = min(N_USERS, len(user_scores))
sampled_users = {u for u, _, _ in user_scores[:n_sample]}

print(f"  Total available users (post 5-core): {len(user_scores):,}")
print(f"  Selected users:                      {len(sampled_users):,}")

# Statistics on the selected sample: users with 1/2/3 categories
cat_counts_selected = defaultdict(int)
for u, n_cat, _ in user_scores[:n_sample]:
    cat_counts_selected[n_cat] += 1
print(f"  n_categories distribution in selected sample: "
      f"{dict(sorted(cat_counts_selected.items()))}")

user_items = {u: user_items[u] for u in sampled_users}
item_users = defaultdict(set)
for user, interactions in user_items.items():
    for _, item in interactions:
        item_users[item].add(user)

# ── STEP 5.5: 5-core reconvergence on selected subset (NEW) ──────────────────
print()
print("=" * 60)
print("STEP 5.5 — Checking/reconverging 5-core on the selected sample...")
print("=" * 60)

iteration = 0
while True:
    iteration += 1
    valid_items = {i for i, u in item_users.items() if len(u) >= MIN_INTERACTIONS}
    new_user_items = {}
    for user, interactions in user_items.items():
        filtered = [(ts, i) for ts, i in interactions if i in valid_items]
        if len(filtered) >= MIN_INTERACTIONS:
            new_user_items[user] = filtered
    new_item_users = defaultdict(set)
    for user, interactions in new_user_items.items():
        for _, item in interactions:
            new_item_users[item].add(user)
    if (len(new_user_items) == len(user_items) and
            len(new_item_users) == len(item_users)):
        print(f"  Convergence confirmed at iteration {iteration} "
              f"(no users/items lost after selection).")
        break
    print(f"  Iteration {iteration}: the sample was not yet 5-core consistent, "
          f"{len(user_items):,} users, {len(item_users):,} items -> recomputing...")
    user_items = new_user_items
    item_users = new_item_users

valid_items_set = set(item_users.keys())

print(f"  Final users: {len(user_items):,}")
print(f"  Final items: {len(valid_items_set):,}")

# ── STEP 6: price and rank discretization — identical to v2 ──────────────────
print()
print("=" * 60)
print("STEP 6 — Discretizing price (quartiles) and rank (tiers)...")
print("=" * 60)

valid_prices = {i: item_price[i] for i in valid_items_set if i in item_price}
if valid_prices:
    price_series = pd.Series(valid_prices)
    price_quartiles = pd.qcut(price_series, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'],
                               duplicates='drop')
    item_price_tier = price_quartiles.to_dict()
else:
    item_price_tier = {}

print(f"  Items with price_tier:      {len(item_price_tier):,} "
      f"({len(item_price_tier)/len(valid_items_set)*100:.1f}%)")

valid_ranks = {i: item_rank[i] for i in valid_items_set if i in item_rank}
if valid_ranks:
    rank_series = pd.Series(valid_ranks)
    rank_tiers  = pd.qcut(rank_series, q=3,
                           labels=['high_popularity', 'mid_popularity', 'low_popularity'],
                           duplicates='drop')
    item_popularity_tier = rank_tiers.to_dict()
else:
    item_popularity_tier = {}

print(f"  Items with popularity_tier: {len(item_popularity_tier):,} "
      f"({len(item_popularity_tier)/len(valid_items_set)*100:.1f}%)")

# ── STEP 6.5: save intermediate validation data ──────────────────────────────
print()
print("=" * 60)
print("STEP 6.5 — Saving intermediate validation data...")
print("=" * 60)

interactions_to_save = []
for user, inter in user_items.items():
    for ts, item in inter:
        interactions_to_save.append({"reviewerID": user, "asin": item, "unixReviewTime": ts})
df_interactions = pd.DataFrame(interactions_to_save)
interactions_path = os.path.join(VALIDATION_DATA_DIR, "interactions_5_core_filtered.csv")
df_interactions.to_csv(interactions_path, index=False)
print(f"  Written: {interactions_path}")

metadata_to_save = {}
for item in valid_items_set:
    metadata_to_save[item] = {
        "categories": item_category.get(item),
        "brand": item_brand.get(item),
        "price": item_price.get(item),
        "price_tier": str(item_price_tier.get(item)) if item in item_price_tier else None,
        "rank": item_rank.get(item),
        "popularity_tier": str(item_popularity_tier.get(item)) if item in item_popularity_tier else None,
    }
metadata_path = os.path.join(VALIDATION_DATA_DIR, "metadata_valid_items.json")
with open(metadata_path, 'w') as f:
    json.dump(metadata_to_save, f)
print(f"  Written: {metadata_path}")

also_buy_to_save = {}
for item_a, ab_list in item_also_buy.items():
    if item_a in valid_items_set:
        filtered_ab = [item_b for item_b in ab_list if item_b in valid_items_set]
        if filtered_ab:
            also_buy_to_save[item_a] = filtered_ab
also_buy_path = os.path.join(VALIDATION_DATA_DIR, "also_buy_pairs.json")
with open(also_buy_path, 'w') as f:
    json.dump(also_buy_to_save, f)
print(f"  Written: {also_buy_path}")

# ── STEP 7: build KG triples — identical to v2 ───────────────────────────────
print()
print("=" * 60)
print("STEP 7 — Building KG triples...")
print("=" * 60)

compatible_with_triples   = []
belongs_to_triples        = []
belongs_to_brand_triples  = []
belongs_to_price_triples  = []
belongs_to_pop_triples    = []

for item in valid_items_set:
    cat = item_category.get(item)
    if cat:
        belongs_to_triples.append((item, cat))

for item in valid_items_set:
    brand = item_brand.get(item)
    if brand:
        brand_clean = brand.replace('\t', ' ').replace('\n', ' ').replace('\r', '').strip()
        if brand_clean:
            belongs_to_brand_triples.append((item, brand_clean))

for item, tier in item_price_tier.items():
    belongs_to_price_triples.append((item, str(tier)))

for item, tier in item_popularity_tier.items():
    belongs_to_pop_triples.append((item, str(tier)))

seen_pairs = set()
for item_a, ab_list in also_buy_to_save.items():
    cat_a = item_category.get(item_a)
    for item_b in ab_list:
        cat_b = item_category.get(item_b)
        if cat_a is None or cat_b is None:
            continue
        if cat_a == cat_b:
            continue
        pair = tuple(sorted([item_a, item_b]))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        compatible_with_triples.append((min(item_a, item_b), max(item_a, item_b)))

print(f"  compatible_with triples:        {len(compatible_with_triples):,}")
print(f"  belongs_to triples:             {len(belongs_to_triples):,}")
print(f"  belongs_to_brand triples:       {len(belongs_to_brand_triples):,}")
print(f"  belongs_to_price_tier triples:  {len(belongs_to_price_triples):,}")
print(f"  belongs_to_pop_tier triples:    {len(belongs_to_pop_triples):,}")

# ── STEP 8: Task A split — identical to v1/v2 ────────────────────────────────
print()
print("=" * 60)
print("STEP 8 — Task A split (90/5/5)...")
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

# ── STEP 9: ID mapping — identical to v2 ─────────────────────────────────────
print()
print("=" * 60)
print("STEP 9 — Building ID mapping...")
print("=" * 60)

all_entities = set()
for item in valid_items_set:
    all_entities.add(f"item_{item}")
for cat in KEEP_CATEGORIES:
    all_entities.add(f"category_{cat}")
all_brands = {b.replace('\t', ' ').replace('\n', ' ').replace('\r', '').strip()
              for b in item_brand.values() if b}
all_brands.discard('')
for brand in all_brands:
    all_entities.add(f"brand_{brand}")

for tier in set(str(t) for t in item_price_tier.values()):
    all_entities.add(f"pricetier_{tier}")

for tier in set(str(t) for t in item_popularity_tier.values()):
    all_entities.add(f"poptier_{tier}")

entity2id = {e: i for i, e in enumerate(sorted(all_entities))}
relation2id = {
    "compatible_with": 0,
    "belongs_to": 1,
    "belongs_to_brand": 2,
    "belongs_to_price_tier": 3,
    "belongs_to_pop_tier": 4,
}

print(f"  Total entities:  {len(entity2id):,}")
print(f"  Total relations: {len(relation2id):,}")

# ── STEP 10: write output files — identical to v2 ────────────────────────────
print()
print("=" * 60)
print("STEP 10 — Writing output files...")
print("=" * 60)


def write_triples(path, triples, relation, head_prefix='item', tail_prefix='item', mode='w'):
    with open(path, mode, encoding="utf-8") as f:
        if mode == 'w':
            f.write("head\trelation\ttail\n")
        for head, tail in triples:
            f.write(f"{head_prefix}_{head}\t{relation}\t{tail_prefix}_{tail}\n")


kg_train_path = os.path.join(OUT_DIR, "kg_train.tsv")
write_triples(kg_train_path, cw_train, "compatible_with", 'item', 'item', mode='w')
write_triples(kg_train_path, belongs_to_triples, "belongs_to", 'item', 'category', mode='a')
write_triples(kg_train_path, belongs_to_brand_triples, "belongs_to_brand", 'item', 'brand', mode='a')
write_triples(kg_train_path, belongs_to_price_triples, "belongs_to_price_tier", 'item', 'pricetier', mode='a')
write_triples(kg_train_path, belongs_to_pop_triples, "belongs_to_pop_tier", 'item', 'poptier', mode='a')
print(f"  Written: kg_train.tsv")

valid_path = os.path.join(OUT_DIR, "taskA_valid.tsv")
write_triples(valid_path, cw_valid, "compatible_with", 'item', 'item', mode='w')
print(f"  Written: taskA_valid.tsv")

test_path = os.path.join(OUT_DIR, "taskA_test.tsv")
write_triples(test_path, cw_test, "compatible_with", 'item', 'item', mode='w')
print(f"  Written: taskA_test.tsv")

e2id_path = os.path.join(OUT_DIR, "entity2id.tsv")
with open(e2id_path, "w", encoding="utf-8") as f:
    f.write("entity\tid\n")
    for e, i in entity2id.items():
        f.write(f"{e}\t{i}\n")
print(f"  Written: entity2id.tsv")

r2id_path = os.path.join(OUT_DIR, "relation2id.tsv")
with open(r2id_path, "w", encoding="utf-8") as f:
    f.write("relation\tid\n")
    for r, i in relation2id.items():
        f.write(f"{r}\t{i}\n")
print(f"  Written: relation2id.tsv")

# ── STEP 11: final statistics ────────────────────────────────────────────────
total_kg_triples = (len(cw_train) + len(belongs_to_triples) +
                     len(belongs_to_brand_triples) + len(belongs_to_price_triples) +
                     len(belongs_to_pop_triples))

avg_interazioni = sum(len(v) for v in user_items.values()) / len(user_items)

stats = f"""
FASHION KG v3 — FINAL STATISTICS
====================================
User selection criterion: decreasing (n_categories, n_interactions)
(NOT random sampling)

Selected users:                  {len(user_items):,}
Average interactions per user:   {avg_interazioni:.2f}
Valid items:                     {len(valid_items_set):,}
Total entities (KG):             {len(entity2id):,}
Relations:                       {len(relation2id):,}

Total compatible_with triples:   {len(compatible_with_triples):,}
  -> train:                      {len(cw_train):,}
  -> valid (Task A):             {len(cw_valid):,}
  -> test  (Task A):             {len(cw_test):,}

belongs_to triples:              {len(belongs_to_triples):,}
belongs_to_brand triples:        {len(belongs_to_brand_triples):,}
belongs_to_price_tier triples:   {len(belongs_to_price_triples):,}
belongs_to_pop_tier triples:     {len(belongs_to_pop_triples):,}
Total KG triples (train):        {total_kg_triples:,}

Seed: {SEED} (used only for the Task A split, not for user selection)
"""

print(stats)
with open(os.path.join(OUT_DIR, "kg_stats.txt"), "w", encoding="utf-8") as f:
    f.write(stats)

print()
print("=" * 60)
print("KG V3 BUILD COMPLETED")
print("=" * 60)
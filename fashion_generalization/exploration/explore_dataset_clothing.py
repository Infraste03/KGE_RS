# fashion_generalization/exploration/explore_dataset_clothing.py

import os
import gzip
import json
from collections import defaultdict

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

# ── Parameters ────────────────────────────────────────────────────────────────
MIN_INTERACTIONS = 5

# Macro-categories extracted from the hierarchy in the 'category' field
CATEGORY_MAP = {
    "top":       ["tops", "tees", "blouses", "shirts", "sweaters",
                  "sweatshirts", "hoodies", "tank", "tunic", "polos"],
    "bottom":    ["pants", "jeans", "shorts", "skirts", "leggings",
                  "trousers", "chinos"],
    "shoes":     ["shoes", "sneakers", "boots", "sandals", "loafers",
                  "heels", "flats", "oxfords", "slippers", "flip-flops"],
    "outerwear": ["jackets", "coats", "blazers", "vests", "parkas",
                  "raincoats"],
    "accessory": ["bags", "belts", "hats", "caps", "scarves",
                  "wallets", "sunglasses", "jewelry", "watches", "gloves"],
    "dress":     ["dresses", "jumpsuits", "rompers", "gowns"],
}

KEEP_CATEGORIES = {"top", "bottom", "shoes"}

def get_macro_category(category_list):
    """
    Given the hierarchical 'category' field, returns the corresponding
    macro-category or None.
    """
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
print("STEP 1 — Reading review file...")
print("=" * 60)

user_items = defaultdict(list)
item_users = defaultdict(set)
total_reviews = 0

with gzip.open(REVIEW_FILE, "rt", encoding="utf-8") as f:
    for line in f:
        try:
            record = json.loads(line.strip())
        except:
            continue
        user = record.get("reviewerID")
        item = record.get("asin")
        ts   = record.get("unixReviewTime", 0)
        if user and item:
            user_items[user].append((ts, item))
            item_users[item].add(user)
            total_reviews += 1
        if total_reviews % 1_000_000 == 0:
            print(f"  ...read {total_reviews:,} reviews")

print(f"  Total reviews:       {total_reviews:,}")
print(f"  Unique users (raw):  {len(user_items):,}")
print(f"  Unique items (raw):  {len(item_users):,}")

# ── STEP 2: read metadata ─────────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 2 — Reading metadata (category and also_buy)...")
print("=" * 60)

item_category      = {}
item_also_buy      = {}
total_meta         = 0
meta_with_cat      = 0
meta_with_also_buy = 0

with gzip.open(META_FILE, "rt", encoding="utf-8") as f:
    for line in f:
        try:
            record = json.loads(line.strip())
        except:
            continue
        total_meta += 1
        item = record.get("asin")
        if not item:
            continue

        # Category
        cats  = record.get("category", [])
        macro = get_macro_category(cats)
        if macro:
            item_category[item] = macro
            meta_with_cat += 1

        # also_buy
        ab = record.get("also_buy")
        if ab:
            item_also_buy[item] = ab
            meta_with_also_buy += 1

        if total_meta % 500_000 == 0:
            print(f"  ...read {total_meta:,} metadata records")

print(f"  Total products in metadata:             {total_meta:,}")
print(f"  With recognized macro-category:         {meta_with_cat:,}")
print(f"  With populated also_buy field:          {meta_with_also_buy:,}")


# ── STEP 3: filter by Fashion category ────────────────────────────────────────
print()
print("=" * 60)
print("STEP 3 — Filtering: keep only items in selected Fashion categories...")
print("=" * 60)

fashion_items = {item for item, cat in item_category.items()
                 if cat in KEEP_CATEGORIES}

# Filter user_items and item_users, keeping only Fashion items
user_items_f = defaultdict(list)
item_users_f = defaultdict(set)

for user, interactions in user_items.items():
    filtered = [(ts, i) for ts, i in interactions
            if i in fashion_items
            and item_category.get(i) in KEEP_CATEGORIES]
    if filtered:
        user_items_f[user] = filtered
        for _, item in filtered:
            item_users_f[item].add(user)

print(f"  Users with at least 1 Fashion item:  {len(user_items_f):,}")
print(f"  Fashion items in sequences:          {len(item_users_f):,}")

# ── STEP 4: iterative 5-core filtering ────────────────────────────────────────
print()
print("=" * 60)
print(f"STEP 4 — Iterative {MIN_INTERACTIONS}-core filtering...")
print("=" * 60)

user_items = user_items_f
item_users = item_users_f
iteration  = 0

while True:
    iteration += 1
    valid_items = {i for i, u in item_users.items()
                   if len(u) >= MIN_INTERACTIONS}
    new_user_items = {}
    for user, interactions in user_items.items():
        filtered = [(ts, i) for ts, i in interactions
                    if i in valid_items]
        if len(filtered) >= MIN_INTERACTIONS:
            new_user_items[user] = filtered

    new_item_users = defaultdict(set)
    for user, interactions in new_user_items.items():
        for _, item in interactions:
            new_item_users[item].add(user)

    if (len(new_user_items) == len(user_items) and
            len(new_item_users) == len(item_users)):
        print(f"  Converged at iteration {iteration}.")
        break

    user_items = new_user_items
    item_users = new_item_users
    print(f"  Iteration {iteration}: "
          f"{len(user_items):,} users, {len(item_users):,} items")

valid_items_set  = set(item_users.keys())
valid_users_set  = set(user_items.keys())
total_inter      = sum(len(v) for v in user_items.values())
seq_lengths      = [len(v) for v in user_items.values()]

print(f"  Final users:              {len(valid_users_set):,}")
print(f"  Final items:              {len(valid_items_set):,}")
print(f"  Total interactions:       {total_inter:,}")
print(f"  Average sequence length:  {sum(seq_lengths)/len(seq_lengths):.1f}")
print(f"  Maximum sequence length:  {max(seq_lengths)}")


# ── STEP 4b: sample 15K users from the already filtered dataset ───────────────
print()
print("=" * 60)
print("STEP 4b — Sampling 15,000 users from the cleaned dataset...")
print("=" * 60)

import random
random.seed(42)

all_valid_users = list(user_items.keys())

# If fewer than 15K users are available, keep all of them
n_sample = min(15_000, len(all_valid_users))
sampled_users = set(random.sample(all_valid_users, n_sample))

# Rebuild user_items and item_users on the sampled users
user_items_s = {u: user_items[u] for u in sampled_users}
item_users_s = defaultdict(set)
for user, interactions in user_items_s.items():
    for _, item in interactions:
        item_users_s[item].add(user)

user_items = user_items_s
item_users = item_users_s
valid_items_set = set(item_users.keys())
valid_users_set = set(user_items.keys())
total_inter = sum(len(v) for v in user_items.values())
seq_lengths = [len(v) for v in user_items.values()]

print(f"  Sampled users:           {len(valid_users_set):,}")
print(f"  Unique items:            {len(valid_items_set):,}")
print(f"  Total interactions:      {total_inter:,}")
print(f"  Average sequence length: {sum(seq_lengths)/len(seq_lengths):.1f}")

# ── STEP 5: KG verification ───────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 5 — Checking cross-category also_buy pairs...")
print("=" * 60)

total_ab       = 0
cross_cat      = 0
same_cat       = 0
unknown_cat    = 0
pair_counts    = defaultdict(int)
items_with_kg  = set()

for item_a, ab_list in item_also_buy.items():
    if item_a not in valid_items_set:
        continue
    cat_a = item_category.get(item_a)
    for item_b in ab_list:
        if item_b not in valid_items_set:
            continue
        total_ab += 1
        cat_b = item_category.get(item_b)
        if cat_a is None or cat_b is None:
            unknown_cat += 1
        elif cat_a == cat_b:
            same_cat += 1
        else:
            cross_cat += 1
            items_with_kg.add(item_a)
            pair_key = "-".join(sorted([cat_a, cat_b]))
            pair_counts[pair_key] += 1

print(f"  Total also_buy pairs (valid items):       {total_ab:,}")
print(f"  → Cross-category pairs (KG):              {cross_cat:,}")
print(f"  → Same-category pairs (discarded):        {same_cat:,}")
print(f"  → Unknown category:                       {unknown_cat:,}")
print(f"  Items with at least one KG pair:          {len(items_with_kg):,}")
print()
print("  Pair-type distribution:")
for pair, count in sorted(pair_counts.items(), key=lambda x: -x[1]):
    print(f"    {pair:<30} {count:>8,}")

# ── STEP 6: review-metadata overlap ───────────────────────────────────────────
print()
print("=" * 60)
print("STEP 6 — Review / metadata overlap...")
print("=" * 60)

items_in_meta    = set(item_category.keys())
overlap          = valid_items_set & items_in_meta
overlap_pct      = len(overlap) / len(valid_items_set) * 100 if valid_items_set else 0
items_with_ab    = valid_items_set & set(item_also_buy.keys())
ab_pct           = len(items_with_ab) / len(valid_items_set) * 100 if valid_items_set else 0

print(f"  Items in sequences (after filtering):  {len(valid_items_set):,}")
print(f"  Also present in metadata:              {len(overlap):,} ({overlap_pct:.1f}%)")
print(f"  With also_buy:                         {len(items_with_ab):,} ({ab_pct:.1f}%)")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"  [SASRec]  Users:                    {len(valid_users_set):,}")
print(f"  [SASRec]  Items:                    {len(valid_items_set):,}")
print(f"  [SASRec]  Interactions:             {total_inter:,}")
print(f"  [SASRec]  Average sequence length:  {sum(seq_lengths)/len(seq_lengths):.1f}")
print(f"  [TransE]  Cross-category KG pairs:  {cross_cat:,}")
print(f"  [KG]      Items covered by KG:      {len(items_with_kg):,}")
print()

# Count the actual macro-category distribution in the metadata
from collections import Counter
cat_distribution = Counter(item_category.values())

print("\nMacro-category distribution in metadata:")
for cat, count in cat_distribution.most_common():
    print(f"  {cat:<15} {count:>10,}")

ok_sasrec = len(valid_users_set) > 1000 and sum(seq_lengths)/len(seq_lengths) >= 5
ok_transe = cross_cat >= 10_000

if ok_sasrec:
    print("  [OK] Dataset suitable for SASRec")
else:
    print("  [!!] Dataset too sparse for SASRec")

if ok_transe:
    print("  [OK] Dataset suitable for TransE / KG")
else:
    print("  [!!] Insufficient cross-category pairs for TransE")

if ok_sasrec and ok_transe:
    print()
    print("  => Dataset SUITABLE for alternate learning.")
else:
    print()
    print("  => Dataset NOT suitable. Review the strategy.")
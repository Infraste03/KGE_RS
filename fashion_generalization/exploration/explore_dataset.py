# fashion_generalization/exploration/explore_dataset.py

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
    "Amazon_Fashion.jsonl.gz"
)

META_FILE = os.path.join(
    FASHION_ROOT,
    "data",
    "raw",
    "meta_Amazon_Fashion.jsonl.gz"
)

OUTPUT_FILE = os.path.join(THIS_DIR, "stats_output.txt")

# ── Parameters ────────────────────────────────────────────────────────────────
MIN_INTERACTIONS = 3   # 3-core threshold

# Main Fashion categories considered in the analysis
# Keys represent the macro-types to recognize in Amazon categories
CATEGORY_MAP = {
    "top":        ["tops", "tees", "blouses", "shirts", "sweaters",
                   "sweatshirts", "hoodies", "tank", "tunic"],
    "bottom":     ["pants", "jeans", "shorts", "skirts", "leggings",
                   "trousers"],
    "shoes":      ["shoes", "sneakers", "boots", "sandals", "loafers",
                   "heels", "flats", "oxfords"],
    "outerwear":  ["jackets", "coats", "blazers", "vests", "parkas"],
    "accessory":  ["bags", "belts", "hats", "caps", "scarves",
                   "wallets", "sunglasses", "jewelry", "watches"],
    "dress":      ["dresses", "jumpsuits", "rompers"],
}


def get_macro_category(categories_list):
    """
    Given the 'categories' field of an item (list of strings),
    returns the corresponding macro-category (top/bottom/shoes/...) or None.
    """
    if not categories_list:
        return None

    # Combine all category strings into a single lowercase text
    text = " ".join(categories_list).lower()

    for macro, keywords in CATEGORY_MAP.items():
        for kw in keywords:
            if kw in text:
                return macro

    return None


def log(lines, text):
    """Prints and stores text for the output file."""
    print(text)
    lines.append(text)


# ── STEP 1: read reviews ──────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Reading review file...")
print("=" * 60)

# user_items[user] = list of (timestamp, parent_asin)
user_items = defaultdict(list)

# item_users[item] = set of users who interacted with it
item_users = defaultdict(set)

total_reviews = 0

with gzip.open(REVIEW_FILE, "rt", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line.strip())

        user = record.get("user_id")
        item = record.get("parent_asin")
        ts   = record.get("timestamp", 0)

        if user and item:
            user_items[user].append((ts, item))
            item_users[item].add(user)
            total_reviews += 1

print(f"  Total reviews read:  {total_reviews:,}")
print(f"  Unique users (raw):  {len(user_items):,}")
print(f"  Unique items (raw):  {len(item_users):,}")

# ── STEP 2: iterative 3-core filtering ────────────────────────────────────────
print()
print("=" * 60)
print("STEP 2 — Iterative 3-core filtering...")
print("=" * 60)

iteration = 0

while True:
    iteration += 1

    # Items with at least MIN_INTERACTIONS valid users
    valid_items = set(
        i for i, u in item_users.items()
        if len(u) >= MIN_INTERACTIONS
    )

    valid_users_new = {}

    for user, interactions in user_items.items():
        filtered = [
            (ts, i)
            for ts, i in interactions
            if i in valid_items
        ]

        if len(filtered) >= MIN_INTERACTIONS:
            valid_users_new[user] = filtered

    # Recompute item_users considering only valid users
    item_users_new = defaultdict(set)

    for user, interactions in valid_users_new.items():
        for _, item in interactions:
            item_users_new[item].add(user)

    # Check convergence
    if (len(valid_users_new) == len(user_items) and
            len(item_users_new) == len(item_users)):
        print(f"  Convergence reached at iteration {iteration}.")
        break

    user_items = valid_users_new
    item_users = item_users_new

    print(f"  Iteration {iteration}: "
          f"{len(user_items):,} users, {len(item_users):,} items")

valid_items_set = set(item_users.keys())
valid_users_set = set(user_items.keys())
total_interactions_after = sum(len(v) for v in user_items.values())

print(f"  Users after 3-core:        {len(valid_users_set):,}")
print(f"  Items after 3-core:        {len(valid_items_set):,}")
print(f"  Interactions after 3-core: {total_interactions_after:,}")

seq_lengths = [len(v) for v in user_items.values()]

print(f"  Average sequence length:   {sum(seq_lengths)/len(seq_lengths):.1f}")
print(f"  Maximum sequence length:   {max(seq_lengths)}")

# ── STEP 3: read metadata ─────────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 3 — Reading metadata (valid items only)...")
print("=" * 60)

# item_category[item] = macro-category (top/bottom/...)
item_category = {}

# item_bought_together[item] = list of ASINs bought together
item_bought_together = defaultdict(list)

items_in_meta  = 0
items_with_cat = 0
items_with_bt  = 0

with gzip.open(META_FILE, "rt", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line.strip())

        item = record.get("parent_asin")

        if item not in valid_items_set:
            continue

        items_in_meta += 1

        # Category
        cats = record.get("categories", [])
        macro = get_macro_category(cats)

        if macro:
            item_category[item] = macro
            items_with_cat += 1

        # bought_together
        bt = record.get("bought_together")

        if bt:
            item_bought_together[item] = bt
            items_with_bt += 1

print(f"  Valid items found in metadata:       {items_in_meta:,}")
print(f"  Items with recognized macro-category:{items_with_cat:,}")
print(f"  Items with non-empty bought_together:{items_with_bt:,}")

# ── STEP 4: compatibility-pair analysis ───────────────────────────────────────
print()
print("=" * 60)
print("STEP 4 — Analyzing cross-category bought_together pairs...")
print("=" * 60)

total_bt_pairs     = 0   # all bought_together pairs
cross_cat_pairs    = 0   # cross-category pairs only
same_cat_pairs     = 0   # same-category pairs
unknown_cat_pairs  = 0   # at least one item has no known category

# Counter by pair type (e.g., top-bottom, top-shoes, ...)
pair_type_counts = defaultdict(int)

for item_a, bt_list in item_bought_together.items():
    cat_a = item_category.get(item_a)

    for item_b in bt_list:
        if item_b not in valid_items_set:
            continue   # item_b did not survive the 3-core filtering

        total_bt_pairs += 1
        cat_b = item_category.get(item_b)

        if cat_a is None or cat_b is None:
            unknown_cat_pairs += 1

        elif cat_a == cat_b:
            same_cat_pairs += 1

        else:
            cross_cat_pairs += 1

            # Sort alphabetically to always obtain the same pair representation
            pair_key = "-".join(sorted([cat_a, cat_b]))
            pair_type_counts[pair_key] += 1

print(f"  Total bought_together pairs (valid items): {total_bt_pairs:,}")
print(f"  → Cross-category (usable for KG):          {cross_cat_pairs:,}")
print(f"  → Same-category (discarded):               {same_cat_pairs:,}")
print(f"  → Unknown category (discarded):            {unknown_cat_pairs:,}")

print()
print("  Cross-category pair distribution by type:")

for pair_type, count in sorted(
        pair_type_counts.items(),
        key=lambda x: -x[1]):
    print(f"    {pair_type:<25} {count:>8,}")

# ── STEP 5: macro-category distribution ───────────────────────────────────────
print()
print("=" * 60)
print("STEP 5 — Macro-category distribution among valid items...")
print("=" * 60)

cat_counts = defaultdict(int)

for item, cat in item_category.items():
    if item in valid_items_set:
        cat_counts[cat] += 1

cat_counts["senza_categoria"] = len(valid_items_set) - items_with_cat

for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
    print(f"  {cat:<20} {count:>8,}")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("SUMMARY — Dataset evaluation for the framework")
print("=" * 60)

lines = []

log(lines, f"Users (3-core):                {len(valid_users_set):,}")
log(lines, f"Items (3-core):                {len(valid_items_set):,}")
log(lines, f"Total interactions:            {total_interactions_after:,}")
log(lines, f"Cross-category KG pairs:       {cross_cat_pairs:,}")
log(lines, "")

if cross_cat_pairs >= 10_000:
    log(lines, "[OK] Dataset suitable — sufficient cross-category pairs.")

elif cross_cat_pairs >= 3_000:
    log(lines, "[WARNING] Dataset borderline — consider expanding the categories.")

else:
    log(lines, "[FAIL] Dataset insufficient — too few cross-category pairs.")

# Save summary to file
with open(OUTPUT_FILE, "w") as out:
    out.write("\n".join(lines))

print(f"\nSummary saved to: {OUTPUT_FILE}")
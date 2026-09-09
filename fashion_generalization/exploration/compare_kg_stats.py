# fashion_generalization/exploration/compare_kg_stats.py

import os
import pandas as pd
from collections import defaultdict

# ── Repository paths ──────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FASHION_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(FASHION_ROOT, ".."))

# ── B2B KG path ───────────────────────────────────────────────────────────────
B2B_KG_PATH = os.path.join(
    REPO_ROOT,
    "B2B_alternate_learning",
    "data",
    "processed",
    "kg_train.tsv"
)

# ── Fashion KG path ───────────────────────────────────────────────────────────
FASHION_KG_PATH = os.path.join(
    FASHION_ROOT,
    "data",
    "processed",
    "kg_train.tsv"
)


def analyze_kg(path, name):
    print(f"\n{'='*50}")
    print(f"KG: {name}")
    print(f"{'='*50}")

    kg = pd.read_csv(path, sep='\t')
    print(f"Total triples:     {len(kg):,}")

    for rel in kg['relation'].unique():
        sub = kg[kg['relation'] == rel]
        print(f"\nRelation: {rel}")
        print(f"  Triples:             {len(sub):,}")

        # Average head degree
        degree = sub.groupby('head').size()
        print(f"  Unique head entities: {len(degree):,}")
        print(f"  Average degree:       {degree.mean():.2f}")
        print(f"  Maximum degree:       {degree.max()}")
        print(f"  Median degree:        {degree.median():.1f}")


analyze_kg(B2B_KG_PATH, "B2B")
analyze_kg(FASHION_KG_PATH, "Fashion")
import os
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
KG_TRAIN = os.path.join(BASE, "data", "processed", "kg_train.tsv")
OUT_DIR  = os.path.join(BASE, "data", "processed", "ablation")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(KG_TRAIN, sep='\t')
print(f"Triple total: {len(df):,}")

variants = {
    "no_brand":            ["belongs_to_brand"],
    "no_category":         ["belongs_to"],
    "no_compatible_with":  ["compatible_with"],
}

for name, exclude in variants.items():
    filtered = df[~df['relation'].isin(exclude)]
    out_path = os.path.join(OUT_DIR, f"kg_train_{name}.tsv")
    filtered.to_csv(out_path, sep='\t', index=False)
    print(f"{name:20s} | excludes {exclude} | {len(df):,} -> {len(filtered):,} | saved in {out_path}")
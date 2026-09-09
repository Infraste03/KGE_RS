from pathlib import Path
import gzip
import json
from collections import defaultdict, Counter
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = Path(__file__).resolve().parent

RAW_REVIEWS = ROOT / "fashion_generalization" / "data" / "raw" / "Clothing_Shoes_and_Jewelry.json.gz"
RAW_META = ROOT / "fashion_generalization" / "data" / "raw" / "meta_Clothing_Shoes_and_Jewelry.json.gz"

SAMPLE_INTER = ROOT / "fashion_generalization" / "data" / "recbole" / "fashion" / "fashion.inter"
OUT_DIR = ANALYSIS_DIR / "fashion_raw_vs_sample"
OUT_DIR.mkdir(parents=True, exist_ok=True)



def find_col(df, prefix):
    for c in df.columns:
        if c == prefix or c.startswith(prefix + ":"):
            return c
    raise ValueError(f"Column {prefix} not found. Columns: {df.columns.tolist()}")


def describe(values):
    s = pd.Series(values)
    return {
        "min": s.min(),
        "q25": s.quantile(0.25),
        "median": s.median(),
        "mean": s.mean(),
        "q75": s.quantile(0.75),
        "q90": s.quantile(0.90),
        "q95": s.quantile(0.95),
        "q99": s.quantile(0.99),
        "max": s.max(),
        "std": s.std(),
    }


def gini(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan
    values += 1e-12
    values = np.sort(values)
    n = len(values)
    return (2 * np.sum(np.arange(1, n + 1) * values) / (n * np.sum(values))) - ((n + 1) / n)


# ============================================================
# 1. ANALISI RAW REVIEW COMPLETO
# ============================================================

def analyze_raw_reviews(path):
    print("\n" + "=" * 80)
    print("ANALYSIS RAW COMPLETE")
    print("=" * 80)

    user_counts = Counter()
    item_counts = Counter()
    user_item_pairs = set()

    n_reviews = 0
    n_missing_user = 0
    n_missing_item = 0
    min_ts = None
    max_ts = None

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue

            user = r.get("reviewerID")
            item = r.get("asin")
            ts = r.get("unixReviewTime")

            if not user:
                n_missing_user += 1
                continue
            if not item:
                n_missing_item += 1
                continue

            n_reviews += 1
            user_counts[user] += 1
            item_counts[item] += 1
            user_item_pairs.add((user, item))

            if ts is not None:
                min_ts = ts if min_ts is None else min(min_ts, ts)
                max_ts = ts if max_ts is None else max(max_ts, ts)

            if n_reviews % 1_000_000 == 0:
                print(f"  lette {n_reviews:,} review valide...")

    n_users = len(user_counts)
    n_items = len(item_counts)
    n_unique_pairs = len(user_item_pairs)
    repeat_interactions = n_reviews - n_unique_pairs

    stats = {
        "dataset": "RAW_FULL",
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_reviews,
        "n_unique_user_item_pairs": n_unique_pairs,
        "repeat_interactions": repeat_interactions,
        "repeat_ratio_pct": repeat_interactions / n_reviews * 100 if n_reviews else 0,
        "sparsity_pct": (1 - n_reviews / (n_users * n_items)) * 100,
        "density_pct": n_reviews / (n_users * n_items) * 100,
        "seq_len": describe(list(user_counts.values())),
        "item_popularity": describe(list(item_counts.values())),
        "seq_len_gini": gini(list(user_counts.values())),
        "item_popularity_gini": gini(list(item_counts.values())),
        "timestamp_min": min_ts,
        "timestamp_max": max_ts,
        "missing_user": n_missing_user,
        "missing_item": n_missing_item,
    }

    return stats, set(user_counts.keys()), set(item_counts.keys())


# ============================================================
# 2. ANALISI SUBSET RECBole
# ============================================================

def analyze_sample_inter(path):
    print("\n" + "=" * 80)
    print("ANALYSIS SUBSET USED FOR RECBole")
    print("=" * 80)

    df = pd.read_csv(path, sep="\t")

    user_col = find_col(df, "user_id")
    item_col = find_col(df, "item_id")
    time_col = find_col(df, "timestamp")

    n_inter = len(df)
    n_users = df[user_col].nunique()
    n_items = df[item_col].nunique()

    user_counts = df.groupby(user_col)[item_col].count()
    item_counts = df.groupby(item_col)[user_col].count()

    n_unique_pairs = df.drop_duplicates([user_col, item_col]).shape[0]
    repeat_interactions = n_inter - n_unique_pairs

    stats = {
        "dataset": "SAMPLE_50K_USERS",
        "n_users": int(n_users),
        "n_items": int(n_items),
        "n_interactions": int(n_inter),
        "n_unique_user_item_pairs": int(n_unique_pairs),
        "repeat_interactions": int(repeat_interactions),
        "repeat_ratio_pct": repeat_interactions / n_inter * 100 if n_inter else 0,
        "sparsity_pct": (1 - n_inter / (n_users * n_items)) * 100,
        "density_pct": n_inter / (n_users * n_items) * 100,
        "seq_len": describe(user_counts.values),
        "item_popularity": describe(item_counts.values),
        "seq_len_gini": gini(user_counts.values),
        "item_popularity_gini": gini(item_counts.values),
        "timestamp_min": df[time_col].min(),
        "timestamp_max": df[time_col].max(),
    }

    return stats, set(df[user_col].astype(str)), set(df[item_col].astype(str))


# ============================================================
# 3. ANALISI METADATA RAW
# ============================================================

def analyze_raw_metadata(path, sample_items):
    print("\n" + "=" * 80)
    print("ANALYSIS OF RAW METADATA")
    print("=" * 80)

    total_meta = 0
    meta_items = set()
    items_with_category = set()
    items_with_also_buy = set()

    sample_items_in_meta = set()
    sample_items_with_category = set()
    sample_items_with_also_buy = set()

    field_counter = Counter()

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue

            total_meta += 1
            for k in r.keys():
                field_counter[k] += 1

            item = r.get("asin")
            if not item:
                continue

            meta_items.add(item)

            if r.get("category"):
                items_with_category.add(item)

            if r.get("also_buy"):
                items_with_also_buy.add(item)

            if item in sample_items:
                sample_items_in_meta.add(item)
                if r.get("category"):
                    sample_items_with_category.add(item)
                if r.get("also_buy"):
                    sample_items_with_also_buy.add(item)

            if total_meta % 500_000 == 0:
                print(f"  letti {total_meta:,} record metadata...")

    stats = {
        "total_meta_records": total_meta,
        "unique_meta_items": len(meta_items),
        "items_with_category": len(items_with_category),
        "items_with_also_buy": len(items_with_also_buy),
        "sample_items": len(sample_items),
        "sample_items_in_meta": len(sample_items_in_meta),
        "sample_items_with_category": len(sample_items_with_category),
        "sample_items_with_also_buy": len(sample_items_with_also_buy),
        "sample_meta_coverage_pct": len(sample_items_in_meta) / len(sample_items) * 100 if sample_items else 0,
        "sample_category_coverage_pct": len(sample_items_with_category) / len(sample_items) * 100 if sample_items else 0,
        "sample_also_buy_coverage_pct": len(sample_items_with_also_buy) / len(sample_items) * 100 if sample_items else 0,
    }

    fields_df = pd.DataFrame([
        {
            "field": k,
            "count": v,
            "coverage_pct": v / total_meta * 100 if total_meta else 0,
        }
        for k, v in field_counter.items()
    ]).sort_values("count", ascending=False)

    return stats, fields_df, meta_items


# ============================================================
# 4. CONFRONTO RAW VS SAMPLE
# ============================================================

def flatten_stats(stats):
    return {
        "dataset": stats["dataset"],
        "n_users": stats["n_users"],
        "n_items": stats["n_items"],
        "n_interactions": stats["n_interactions"],
        "sparsity_pct": stats["sparsity_pct"],
        "density_pct": stats["density_pct"],
        "repeat_ratio_pct": stats["repeat_ratio_pct"],
        "seq_len_mean": stats["seq_len"]["mean"],
        "seq_len_median": stats["seq_len"]["median"],
        "seq_len_max": stats["seq_len"]["max"],
        "seq_len_gini": stats["seq_len_gini"],
        "item_pop_mean": stats["item_popularity"]["mean"],
        "item_pop_median": stats["item_popularity"]["median"],
        "item_pop_max": stats["item_popularity"]["max"],
        "item_popularity_gini": stats["item_popularity_gini"],
    }


def main():
    raw_stats, raw_users, raw_items = analyze_raw_reviews(RAW_REVIEWS)
    sample_stats, sample_users, sample_items = analyze_sample_inter(SAMPLE_INTER)
    meta_stats, fields_df, meta_items = analyze_raw_metadata(RAW_META, sample_items)

    comparison_df = pd.DataFrame([
        flatten_stats(raw_stats),
        flatten_stats(sample_stats),
    ])

    coverage = {
        "sample_users": len(sample_users),
        "raw_users": len(raw_users),
        "sample_users_pct_of_raw": len(sample_users) / len(raw_users) * 100 if raw_users else 0,

        "sample_items": len(sample_items),
        "raw_items": len(raw_items),
        "sample_items_pct_of_raw_reviews": len(sample_items) / len(raw_items) * 100 if raw_items else 0,

        "sample_interactions": sample_stats["n_interactions"],
        "raw_interactions": raw_stats["n_interactions"],
        "sample_interactions_pct_of_raw": sample_stats["n_interactions"] / raw_stats["n_interactions"] * 100,

        "sample_items_in_metadata_pct": meta_stats["sample_meta_coverage_pct"],
        "sample_items_with_category_pct": meta_stats["sample_category_coverage_pct"],
        "sample_items_with_also_buy_pct": meta_stats["sample_also_buy_coverage_pct"],
    }

    coverage_df = pd.DataFrame([coverage])
    metadata_df = pd.DataFrame([meta_stats])

    comparison_df.to_csv(OUT_DIR / "raw_vs_sample_summary.csv", index=False)
    coverage_df.to_csv(OUT_DIR / "sample_coverage.csv", index=False)
    metadata_df.to_csv(OUT_DIR / "metadata_summary.csv", index=False)
    fields_df.to_csv(OUT_DIR / "metadata_fields.csv", index=False)


    print("\n" + "=" * 80)
    print("CONFRONTO RAW VS SAMPLE")
    print("=" * 80)
    print(comparison_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("COPERTURA DEL SUBSET")
    print("=" * 80)
    print(coverage_df.to_string(index=False))

    print("\nFile salvati in:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()

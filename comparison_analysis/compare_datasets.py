from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# PATH CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = Path(__file__).resolve().parent

PATHS = {
    "B2B": {
        "inter": ROOT / "dataset" / "b2b_data" / "b2b_data.inter",
        "kg_files": [
            ROOT / "data" / "processed" / "kg_train.tsv",
            ROOT / "data" / "processed" / "taskA_valid.tsv",
            ROOT / "data" / "processed" / "taskA_test.tsv",
        ],
    },
    "B2C_Fashion": {
        "inter": ROOT / "fashion_generalization" / "data" / "recbole" / "fashion" / "fashion.inter",
        "kg_files": [
            ROOT / "fashion_generalization" / "data" / "processed" / "kg_train.tsv",
            ROOT / "fashion_generalization" / "data" / "processed" / "taskA_valid.tsv",
            ROOT / "fashion_generalization" / "data" / "processed" / "taskA_test.tsv",
        ],
    },
}

OUT_DIR = ANALYSIS_DIR / "b2b_vs_fashion"
OUT_DIR.mkdir(parents=True, exist_ok=True)



# ============================================================
# HELPERS
# ============================================================

def find_col(df, prefix):
    for col in df.columns:
        if col == prefix or col.startswith(prefix + ":"):
            return col
    raise ValueError(f"Columns '{prefix}' not found. Available columns: {df.columns.tolist()}")


def safe_pct(x, y):
    return round((x / y) * 100, 4) if y else 0.0


def gini(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan
    if np.amin(values) < 0:
        values -= np.amin(values)
    values += 1e-12
    values = np.sort(values)
    n = len(values)
    return round((2 * np.sum((np.arange(1, n + 1) * values)) / (n * np.sum(values))) - ((n + 1) / n), 4)


def describe_series(s):
    return {
        "min": float(s.min()),
        "q25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "mean": float(s.mean()),
        "q75": float(s.quantile(0.75)),
        "q90": float(s.quantile(0.90)),
        "q95": float(s.quantile(0.95)),
        "q99": float(s.quantile(0.99)),
        "max": float(s.max()),
        "std": float(s.std()) if len(s) > 1 else 0.0,
    }


def entity_prefix(x):
    x = str(x)
    return x.split("_", 1)[0] if "_" in x else "no_prefix"


# ============================================================
# INTERACTION DATASET ANALYSIS
# ============================================================

def analyze_inter(path, dataset_name):
    if not path.exists():
        raise FileNotFoundError(f"File .inter not found: {path}")

    df = pd.read_csv(path, sep="\t")

    user_col = find_col(df, "user_id")
    item_col = find_col(df, "item_id")
    time_col = find_col(df, "timestamp")

    n_inter = len(df)
    n_users = df[user_col].nunique()
    n_items = df[item_col].nunique()
    sparsity = 1 - (n_inter / (n_users * n_items))

    seq_len = df.groupby(user_col)[item_col].count()
    item_pop = df.groupby(item_col)[user_col].count()

    repeat_interactions = df.duplicated(subset=[user_col, item_col], keep="first").sum()
    repeat_ratio = repeat_interactions / n_inter if n_inter else 0

    df_time = df.copy()
    df_time[time_col] = pd.to_numeric(df_time[time_col], errors="coerce")
    min_ts = df_time[time_col].min()
    max_ts = df_time[time_col].max()

    same_timestamp_baskets = (
        df.groupby([user_col, time_col])[item_col]
        .count()
        .reset_index(name="basket_size")
    )
    basket_sizes = same_timestamp_baskets["basket_size"]

    stats = {
        "dataset": dataset_name,
        "inter_path": str(path),
        "n_users": int(n_users),
        "n_items": int(n_items),
        "n_interactions": int(n_inter),
        "sparsity_pct": round(sparsity * 100, 6),
        "density_pct": round((1 - sparsity) * 100, 6),
        "repeat_interactions": int(repeat_interactions),
        "repeat_ratio_pct": round(repeat_ratio * 100, 4),
        "seq_len": describe_series(seq_len),
        "item_popularity": describe_series(item_pop),
        "seq_len_gini": gini(seq_len.values),
        "item_popularity_gini": gini(item_pop.values),
        "timestamp_min": float(min_ts),
        "timestamp_max": float(max_ts),
        "n_user_timestamp_baskets": int(len(same_timestamp_baskets)),
        "basket_size": describe_series(basket_sizes),
        "multi_item_basket_pct": safe_pct((basket_sizes > 1).sum(), len(basket_sizes)),
    }

    return stats, df, user_col, item_col


# ============================================================
# KG ANALYSIS
# ============================================================

def load_kg_files(paths):
    existing = [p for p in paths if p.exists()]
    if not existing:
        raise FileNotFoundError(f"No KG files found among: {paths}")

    dfs = []
    for p in existing:
        tmp = pd.read_csv(p, sep="\t")
        tmp["split_file"] = p.name
        dfs.append(tmp)

    kg = pd.concat(dfs, ignore_index=True)

    required = {"head", "relation", "tail"}
    if not required.issubset(kg.columns):
        raise ValueError(f"KG without expected columns {required}. Columns: {kg.columns.tolist()}")

    return kg


def analyze_kg(paths, dataset_name):
    kg = load_kg_files(paths)

    entities = set(kg["head"].astype(str)) | set(kg["tail"].astype(str))
    relations = kg["relation"].astype(str).unique()

    general = {
        "dataset": dataset_name,
        "kg_files_used": [str(p) for p in paths if p.exists()],
        "n_triples": int(len(kg)),
        "n_entities": int(len(entities)),
        "n_relations": int(len(relations)),
        "n_duplicate_triples": int(kg.duplicated(subset=["head", "relation", "tail"]).sum()),
        "n_heads": int(kg["head"].nunique()),
        "n_tails": int(kg["tail"].nunique()),
    }

    prefix_rows = []
    for ent in entities:
        prefix_rows.append({"dataset": dataset_name, "entity_type": entity_prefix(ent), "entity": ent})
    prefix_df = pd.DataFrame(prefix_rows)
    entity_type_counts = (
        prefix_df.groupby("entity_type")["entity"]
        .nunique()
        .reset_index(name="n_entities")
        .sort_values("n_entities", ascending=False)
    )

    rel_rows = []
    for rel, sub in kg.groupby("relation"):
        head_deg = sub.groupby("head").size()
        tail_deg = sub.groupby("tail").size()

        rel_rows.append({
            "dataset": dataset_name,
            "relation": rel,
            "n_triples": int(len(sub)),
            "pct_triples": round(len(sub) / len(kg) * 100, 4),
            "n_unique_heads": int(sub["head"].nunique()),
            "n_unique_tails": int(sub["tail"].nunique()),
            "head_type_mode": sub["head"].astype(str).map(entity_prefix).mode().iloc[0],
            "tail_type_mode": sub["tail"].astype(str).map(entity_prefix).mode().iloc[0],
            "head_degree_mean": round(head_deg.mean(), 4),
            "head_degree_median": round(head_deg.median(), 4),
            "head_degree_max": int(head_deg.max()),
            "tail_degree_mean": round(tail_deg.mean(), 4),
            "tail_degree_median": round(tail_deg.median(), 4),
            "tail_degree_max": int(tail_deg.max()),
            "head_degree_gini": gini(head_deg.values),
            "tail_degree_gini": gini(tail_deg.values),
        })

    rel_stats = pd.DataFrame(rel_rows).sort_values("n_triples", ascending=False)

    split_stats = (
        kg.groupby("split_file")
        .size()
        .reset_index(name="n_triples")
        .sort_values("split_file")
    )
    split_stats.insert(0, "dataset", dataset_name)

    return general, rel_stats, entity_type_counts, split_stats, kg


# ============================================================
# CROSS ANALYSIS: INTERACTION ITEMS VS KG ITEMS
# ============================================================

def analyze_inter_kg_overlap(dataset_name, inter_df, item_col, kg):
    inter_items = set(inter_df[item_col].astype(str))

    kg_entities = set(kg["head"].astype(str)) | set(kg["tail"].astype(str))
    kg_items_prefixed = {x for x in kg_entities if x.startswith("item_")}
    kg_items_raw = {x.replace("item_", "", 1) for x in kg_items_prefixed}

    overlap = inter_items & kg_items_raw

    return {
        "dataset": dataset_name,
        "n_items_inter": int(len(inter_items)),
        "n_items_kg_with_prefix": int(len(kg_items_prefixed)),
        "n_items_overlap_inter_kg": int(len(overlap)),
        "pct_inter_items_covered_by_kg": safe_pct(len(overlap), len(inter_items)),
        "pct_kg_items_present_in_inter": safe_pct(len(overlap), len(kg_items_raw)),
        "items_only_in_inter": int(len(inter_items - kg_items_raw)),
        "items_only_in_kg": int(len(kg_items_raw - inter_items)),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    inter_summary = []
    kg_summary = []
    overlap_summary = []

    all_relation_stats = []
    all_entity_type_stats = []
    all_split_stats = []

    for name, cfg in PATHS.items():
        print(f"\n{'=' * 80}")
        print(f"ANALISI DATASET: {name}")
        print(f"{'=' * 80}")

        inter_stats, inter_df, user_col, item_col = analyze_inter(cfg["inter"], name)
        kg_general, rel_stats, entity_type_counts, split_stats, kg = analyze_kg(cfg["kg_files"], name)
        overlap_stats = analyze_inter_kg_overlap(name, inter_df, item_col, kg)

        inter_summary.append(inter_stats)
        kg_summary.append(kg_general)
        overlap_summary.append(overlap_stats)

        all_relation_stats.append(rel_stats)
        all_entity_type_stats.append(entity_type_counts.assign(dataset=name))
        all_split_stats.append(split_stats)

        print("\n[INTERACTIONS]")
        print(f"Utenti:        {inter_stats['n_users']:,}")
        print(f"Item:          {inter_stats['n_items']:,}")
        print(f"Interazioni:   {inter_stats['n_interactions']:,}")
        print(f"Sparsity:      {inter_stats['sparsity_pct']}%")
        print(f"Seq mean/max:  {inter_stats['seq_len']['mean']:.2f} / {inter_stats['seq_len']['max']:.0f}")
        print(f"Repeat ratio:  {inter_stats['repeat_ratio_pct']}%")

        print("\n[KG]")
        print(f"Entità:        {kg_general['n_entities']:,}")
        print(f"Relazioni:     {kg_general['n_relations']:,}")
        print(f"Triple:        {kg_general['n_triples']:,}")
        print(f"Duplicati:     {kg_general['n_duplicate_triples']:,}")

        print("\n[OVERLAP ITEM INTER-KG]")
        print(f"Item inter coperti dal KG: {overlap_stats['pct_inter_items_covered_by_kg']}%")

    relation_stats_df = pd.concat(all_relation_stats, ignore_index=True)
    entity_type_stats_df = pd.concat(all_entity_type_stats, ignore_index=True)
    split_stats_df = pd.concat(all_split_stats, ignore_index=True)

    inter_flat = []
    for s in inter_summary:
        row = {
            "dataset": s["dataset"],
            "n_users": s["n_users"],
            "n_items": s["n_items"],
            "n_interactions": s["n_interactions"],
            "sparsity_pct": s["sparsity_pct"],
            "density_pct": s["density_pct"],
            "repeat_ratio_pct": s["repeat_ratio_pct"],
            "seq_len_mean": s["seq_len"]["mean"],
            "seq_len_median": s["seq_len"]["median"],
            "seq_len_max": s["seq_len"]["max"],
            "seq_len_gini": s["seq_len_gini"],
            "item_pop_mean": s["item_popularity"]["mean"],
            "item_pop_median": s["item_popularity"]["median"],
            "item_pop_max": s["item_popularity"]["max"],
            "item_popularity_gini": s["item_popularity_gini"],
            "multi_item_basket_pct": s["multi_item_basket_pct"],
        }
        inter_flat.append(row)

    inter_summary_df = pd.DataFrame(inter_flat)
    kg_summary_df = pd.DataFrame(kg_summary)
    overlap_summary_df = pd.DataFrame(overlap_summary)

    inter_summary_df.to_csv(OUT_DIR / "interaction_summary.csv", index=False)
    kg_summary_df.to_csv(OUT_DIR / "kg_summary.csv", index=False)
    relation_stats_df.to_csv(OUT_DIR / "kg_relation_stats.csv", index=False)
    entity_type_stats_df.to_csv(OUT_DIR / "kg_entity_type_stats.csv", index=False)
    split_stats_df.to_csv(OUT_DIR / "kg_split_stats.csv", index=False)
    overlap_summary_df.to_csv(OUT_DIR / "inter_kg_overlap.csv", index=False)

    full_json = {
        "interaction_summary": inter_summary,
        "kg_summary": kg_summary,
        "overlap_summary": overlap_summary,
    }

    with open(OUT_DIR / "comparison_full_stats.json", "w", encoding="utf-8") as f:
        json.dump(full_json, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 80}")
    print("FILE saved IN:")
    print(OUT_DIR)
    print("=" * 80)
    print("interaction_summary.csv")
    print("kg_summary.csv")
    print("kg_relation_stats.csv")
    print("kg_entity_type_stats.csv")
    print("kg_split_stats.csv")
    print("inter_kg_overlap.csv")
    print("comparison_full_stats.json")


if __name__ == "__main__":
    main()
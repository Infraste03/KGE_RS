"""
Knowledge Graph construction script for B2B-Parts-Rec.

Builds the KG from train.csv only, performs preliminary data validation,
extracts 5 relation types, splits compatible_with triples 90/5/5 for Task A
evaluation, and runs post-construction sanity checks.

Inputs:
    dataset/train.csv  -- training transactions (LOO already applied upstream)

Outputs:
    data/processed/kg_train.tsv       -- KG triples for TransE training
    data/processed/taskA_valid.tsv    -- holdout triples for Task A validation
    data/processed/taskA_test.tsv     -- holdout triples for Task A test
    data/processed/kg_stats.json      -- full statistics report
"""

import os
import json
import logging
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Column names (centralized so they are easy to change)
COL_CLIENT = 'CUSTOMER_ID'
COL_MACHINE = 'MACHINE_ID'
COL_MODEL = 'PRODUCT_MODEL_ID'
COL_ITEM = 'ITEM_ID'
COL_DATE = 'REQUEST_DATE'

# Task A split ratios
TASKA_TRAIN_RATIO = 0.90
TASKA_VALID_RATIO = 0.05
TASKA_TEST_RATIO = 0.05

# Threshold for compatible_with_model: an item is compatible with a model
# only if observed with at least K distinct machines of that model.
COMPATIBLE_WITH_MODEL_THRESHOLD = 2 #*test both K=1 and K=2 to see impact on KG size and stats*

# Random seed for reproducibility of the Task A split
RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# Phase 1: Loading and preliminary validation
# --------------------------------------------------------------------------- #

def load_and_validate(train_path):
    """Load the training CSV and run preliminary sanity checks."""
    logger.info("=" * 70)
    logger.info("PHASE 1: Loading and preliminary validation")
    logger.info("=" * 70)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")

    df = pd.read_csv(train_path, sep=';')
    logger.info(f"Loaded {len(df):,} rows from {train_path}")

    # Check that required columns are present
    required_cols = [COL_CLIENT, COL_MACHINE, COL_MODEL, COL_ITEM]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Report missing values for each key column
    logger.info("\nMissing values per key column:")
    for col in required_cols:
        n_missing = df[col].isna().sum()
        pct = 100 * n_missing / len(df)
        logger.info(f"  {col}: {n_missing:,} ({pct:.2f}%)")

    # Detect hidden nullables: empty strings, the string "NaN", whitespace-only
    logger.info("\nHidden-null detection (empty/whitespace strings):")
    for col in required_cols:
        if df[col].dtype == object:
            suspicious = df[col].astype(str).str.strip().isin(['', 'NaN', 'nan', 'None'])
            n_susp = suspicious.sum()
            if n_susp > 0:
                logger.warning(f"  {col}: {n_susp:,} suspicious string values")

    # Distinct entity counts
    logger.info("\nDistinct entity counts:")
    logger.info(f"  Customers : {df[COL_CLIENT].nunique():,}")
    logger.info(f"  Machines  : {df[COL_MACHINE].nunique():,}")
    logger.info(f"  Models    : {df[COL_MODEL].nunique():,}")
    logger.info(f"  Items     : {df[COL_ITEM].nunique():,}")

    # Distribution of transactions per customer (to spot dominant clients)
    txn_per_client = df.groupby(COL_CLIENT).size()
    logger.info("\nTransactions per customer:")
    logger.info(f"  mean   : {txn_per_client.mean():.1f}")
    logger.info(f"  median : {txn_per_client.median():.1f}")
    logger.info(f"  max    : {txn_per_client.max():,}")
    logger.info(f"  min    : {txn_per_client.min():,}")

    return df

# --------------------------------------------------------------------------- #
# Phase 2: Data cleaning
# --------------------------------------------------------------------------- #

def clean_machine_model_inconsistencies(df):
    """
    Verify and fix the assumption that each machine maps to exactly one model.

    If a machine is associated with multiple models, we keep the most frequent
    association (majority vote) and log a warning.
    """
    logger.info("=" * 70)
    logger.info("PHASE 2: Cleaning machine -> model inconsistencies")
    logger.info("=" * 70)

    # Drop rows with missing machine or model for this check
    sub = df[[COL_MACHINE, COL_MODEL]].dropna()

    machine_model_count = sub.groupby(COL_MACHINE)[COL_MODEL].nunique()
    problematic = machine_model_count[machine_model_count > 1]

    if len(problematic) == 0:
        logger.info("All machines map to exactly one model. No cleaning needed.")
        return df

    logger.warning(f"Found {len(problematic):,} machines with multiple models")
    logger.warning(f"Affected machines (sample): {list(problematic.index[:5])}")

    # Majority-vote resolution: pick most frequent model per machine
    canonical = (
        sub.groupby([COL_MACHINE, COL_MODEL])
           .size()
           .reset_index(name='count')
           .sort_values(['count'], ascending=False)
           .drop_duplicates(COL_MACHINE, keep='first')
           [[COL_MACHINE, COL_MODEL]]
           .rename(columns={COL_MODEL: '_canonical_model'})
    )

    df = df.merge(canonical, on=COL_MACHINE, how='left')

    # Flag rows where the model in the row disagrees with the canonical mapping
    n_mismatched = (df[COL_MODEL] != df['_canonical_model']).sum()
    logger.info(f"Rewriting {n_mismatched:,} rows to use canonical model")

    df[COL_MODEL] = df['_canonical_model']
    df = df.drop(columns=['_canonical_model'])
    return df

# --------------------------------------------------------------------------- #
# Phase 3: Extract the four "complete" relations (no holdout split)
# --------------------------------------------------------------------------- #

def make_triples(heads, relation, tails):
    """Build a DataFrame of triples in (head, relation, tail) format."""
    return pd.DataFrame({
        'head': heads.values,
        'relation': relation,
        'tail': tails.values,
    })

def extract_complete_relations(df):
    """
    Extract the four relations that go fully into the KG:
      - (client, owns, machine)
      - (machine, instance_of, model)
      - (client, bought, item)
      - (item, compatible_with_model, model)   [with k>=2 threshold]
    """
    logger.info("=" * 70)
    logger.info("PHASE 3: Extracting complete relations")
    logger.info("=" * 70)

    triples_list = []

    # ----- owns: (client, owns, machine) -----
    sub = df[[COL_CLIENT, COL_MACHINE]].dropna().drop_duplicates()
    heads = 'client_' + sub[COL_CLIENT].astype(str)
    tails = 'machine_' + sub[COL_MACHINE].astype(str)
    t_owns = make_triples(heads, 'owns', tails)
    logger.info(f"  owns                  : {len(t_owns):,} triples")
    triples_list.append(t_owns)

    # ----- instance_of: (machine, instance_of, model) -----
    sub = df[[COL_MACHINE, COL_MODEL]].dropna().drop_duplicates()
    heads = 'machine_' + sub[COL_MACHINE].astype(str)
    tails = 'model_' + sub[COL_MODEL].astype(str)
    t_inst = make_triples(heads, 'instance_of', tails)
    logger.info(f"  instance_of           : {len(t_inst):,} triples")
    triples_list.append(t_inst)

    # ----- bought: (client, bought, item) -----
    sub = df[[COL_CLIENT, COL_ITEM]].dropna().drop_duplicates()
    heads = 'client_' + sub[COL_CLIENT].astype(str)
    tails = 'item_' + sub[COL_ITEM].astype(str)
    t_bought = make_triples(heads, 'bought', tails)
    logger.info(f"  bought                : {len(t_bought):,} triples")
    triples_list.append(t_bought)

    # ----- compatible_with_model: with threshold k>=2 -----
    # An item is compatible with a model only if observed with at least
    # COMPATIBLE_WITH_MODEL_THRESHOLD distinct machines of that model.
    sub = df[[COL_ITEM, COL_MODEL, COL_MACHINE]].dropna().drop_duplicates()
    grouped = sub.groupby([COL_ITEM, COL_MODEL])[COL_MACHINE].nunique()
    valid_pairs = grouped[grouped >= COMPATIBLE_WITH_MODEL_THRESHOLD].reset_index()
    heads = 'item_' + valid_pairs[COL_ITEM].astype(str)
    tails = 'model_' + valid_pairs[COL_MODEL].astype(str)
    t_cwm = make_triples(heads, 'compatible_with_model', tails)
    logger.info(
        f"  compatible_with_model : {len(t_cwm):,} triples "
        f"(threshold k>={COMPATIBLE_WITH_MODEL_THRESHOLD})"
    )
    triples_list.append(t_cwm)

    return pd.concat(triples_list, ignore_index=True)


# --------------------------------------------------------------------------- #
# Phase 4: Extract and split compatible_with for Task A
# --------------------------------------------------------------------------- #

def extract_and_split_compatible_with(df, seed=RANDOM_SEED):
    """
    Extract all distinct (item, machine) pairs as compatible_with triples,
    then split 90/5/5 for Task A evaluation.

    Note: validation/test triples are NOT added to the KG. They are kept
    aside as evaluation queries.
    """
    logger.info("=" * 70)
    logger.info("PHASE 4: Extracting and splitting compatible_with (Task A)")
    logger.info("=" * 70)

    sub = df[[COL_ITEM, COL_MACHINE]].dropna().drop_duplicates().reset_index(drop=True)
    logger.info(f"  total compatible_with pairs : {len(sub):,}")

    heads = 'item_' + sub[COL_ITEM].astype(str)
    tails = 'machine_' + sub[COL_MACHINE].astype(str)
    triples = make_triples(heads, 'compatible_with', tails)

    # Shuffle deterministically
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(triples))
    triples = triples.iloc[perm].reset_index(drop=True)

    n_total = len(triples)
    n_train = int(n_total * TASKA_TRAIN_RATIO)
    n_valid = int(n_total * TASKA_VALID_RATIO)
    # test gets the remainder to avoid rounding loss
    n_test = n_total - n_train - n_valid

    train_part = triples.iloc[:n_train].reset_index(drop=True)
    valid_part = triples.iloc[n_train:n_train + n_valid].reset_index(drop=True)
    test_part = triples.iloc[n_train + n_valid:].reset_index(drop=True)

    logger.info(f"  -> train : {len(train_part):,} (in KG)")
    logger.info(f"  -> valid : {len(valid_part):,} (Task A holdout)")
    logger.info(f"  -> test  : {len(test_part):,} (Task A holdout)")

    return train_part, valid_part, test_part

# --------------------------------------------------------------------------- #
# Phase 5: Post-construction validation
# --------------------------------------------------------------------------- #

def validate_kg(kg_df, taskA_valid, taskA_test):
    """
    Run sanity checks on the final KG:
      - no duplicates
      - leakage check: Task A valid/test triples not in KG
      - transductive setting: all entities in valid/test must be in the KG
    """
    logger.info("=" * 70)
    logger.info("PHASE 5: Post-construction validation")
    logger.info("=" * 70)

    # Duplicate check
    n_dup = kg_df.duplicated().sum()
    if n_dup > 0:
        logger.warning(f"Found {n_dup:,} duplicate triples in KG, dropping")
        kg_df = kg_df.drop_duplicates().reset_index(drop=True)
    else:
        logger.info("No duplicates in KG.")

    # Leakage check: Task A valid/test must NOT appear in KG
    kg_set = set(map(tuple, kg_df.values))

    valid_leaks = sum(1 for r in taskA_valid.values if tuple(r) in kg_set)
    test_leaks = sum(1 for r in taskA_test.values if tuple(r) in kg_set)

    if valid_leaks > 0 or test_leaks > 0:
        raise RuntimeError(
            f"LEAKAGE DETECTED: {valid_leaks} valid + {test_leaks} test triples "
            f"are present in the KG. This must be fixed before training."
        )
    logger.info("No Task A leakage: valid/test triples are disjoint from KG.")

    # Transductive setting check: all entities in valid/test must appear in KG
    kg_entities = set(kg_df['head']).union(set(kg_df['tail']))

    def coverage(name, df_eval):
        ents = set(df_eval['head']).union(set(df_eval['tail']))
        missing = ents - kg_entities
        n_missing_triples = sum(
            1 for r in df_eval.values
            if r[0] not in kg_entities or r[2] not in kg_entities
        )
        logger.info(
            f"  Task A {name}: {len(missing):,} entities not in KG "
            f"-> {n_missing_triples:,} unevaluable triples"
        )
        return n_missing_triples

    n_unev_valid = coverage('valid', taskA_valid)
    n_unev_test = coverage('test', taskA_test)

    return kg_df, n_unev_valid, n_unev_test


def compute_kg_stats(kg_df):
    """Compute structural statistics on the final KG."""
    logger.info("=" * 70)
    logger.info("PHASE 6: KG statistics")
    logger.info("=" * 70)

    stats = {}

    # Triples per relation
    rel_counts = kg_df['relation'].value_counts().to_dict()
    stats['triples_per_relation'] = rel_counts
    logger.info("\nTriples per relation:")
    for rel, count in rel_counts.items():
        logger.info(f"  {rel:25s} : {count:,}")

    # Nodes per type (extracted from prefix)
    all_nodes = pd.concat([kg_df['head'], kg_df['tail']]).unique()
    node_types = {}
    for n in all_nodes:
        prefix = n.split('_')[0]
        node_types[prefix] = node_types.get(prefix, 0) + 1
    stats['nodes_per_type'] = node_types
    logger.info("\nNodes per type:")
    for t, c in sorted(node_types.items()):
        logger.info(f"  {t:25s} : {c:,}")
    logger.info(f"  {'TOTAL':25s} : {len(all_nodes):,}")

    # Degree distribution (head + tail occurrences)
    degree = pd.concat([kg_df['head'], kg_df['tail']]).value_counts()
    stats['degree'] = {
        'mean': float(degree.mean()),
        'median': float(degree.median()),
        'min': int(degree.min()),
        'max': int(degree.max()),
        'p95': float(degree.quantile(0.95)),
    }
    logger.info("\nDegree distribution:")
    for k, v in stats['degree'].items():
        logger.info(f"  {k:6s} : {v}")

    # Top-10 highest-degree nodes (useful to spot anomalous hubs)
    top10 = degree.head(10).to_dict()
    stats['top10_degree'] = top10
    logger.info("\nTop-10 highest-degree nodes:")
    for n, d in top10.items():
        logger.info(f"  {n:30s} : {d:,}")

    return stats

# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #

def main(train_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df = load_and_validate(train_path)
    df = clean_machine_model_inconsistencies(df)

    kg_complete = extract_complete_relations(df)
    cw_train, cw_valid, cw_test = extract_and_split_compatible_with(df)

    # Merge compatible_with train portion into the final KG
    kg_df = pd.concat([kg_complete, cw_train], ignore_index=True)
    kg_df = kg_df.drop_duplicates().reset_index(drop=True)

    kg_df, n_unev_v, n_unev_t = validate_kg(kg_df, cw_valid, cw_test)
    stats = compute_kg_stats(kg_df)

    # Save outputs
    kg_path = os.path.join(output_dir, 'kg_train.tsv')
    valid_path = os.path.join(output_dir, 'taskA_valid.tsv')
    test_path = os.path.join(output_dir, 'taskA_test.tsv')
    stats_path = os.path.join(output_dir, 'kg_stats.json')

    kg_df.to_csv(kg_path, sep='\t', index=False)
    cw_valid.to_csv(valid_path, sep='\t', index=False)
    cw_test.to_csv(test_path, sep='\t', index=False)

    stats['unevaluable_triples_valid'] = n_unev_v
    stats['unevaluable_triples_test'] = n_unev_t
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    logger.info("=" * 70)
    logger.info("DONE")
    logger.info("=" * 70)
    logger.info(f"  KG          -> {kg_path}")
    logger.info(f"  Task A val  -> {valid_path}")
    logger.info(f"  Task A test -> {test_path}")
    logger.info(f"  Stats       -> {stats_path}")


if __name__ == "__main__":
    TRAIN_CSV = os.path.join("dataset", "train.csv")
    OUTPUT_DIR = os.path.join("data", "processed")
    main(TRAIN_CSV, OUTPUT_DIR)
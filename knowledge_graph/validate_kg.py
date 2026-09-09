"""
KG validation script.

Runs after build_kg.py to verify the semantic correctness of the generated
knowledge graph against the original CSV.

Two layers of validation:

  1. Sampled spot checks: pick random triples from the KG and verify they
     correspond to actual rows in the CSV.

  2. Global invariants: properties that must hold across the whole graph,
     such as "each machine has exactly one model" or "compatible_with_model
     triples respect the threshold".
"""

import os
import logging
import random
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ---- Paths ---------------------------------------------------------------- #
TRAIN_CSV = os.path.join("dataset", "train.csv")
KG_PATH = os.path.join("data", "processed", "kg_train.tsv")
TASKA_VALID = os.path.join("data", "processed", "taskA_valid.tsv")
TASKA_TEST = os.path.join("data", "processed", "taskA_test.tsv")

# ---- Config --------------------------------------------------------------- #
COL_CLIENT = 'CUSTOMER_ID'
COL_MACHINE = 'MACHINE_ID'
COL_MODEL = 'PRODUCT_MODEL_ID'
COL_ITEM = 'ITEM_ID'

N_SAMPLES_PER_RELATION = 20  # how many triples to spot-check per relation
COMPATIBLE_WITH_MODEL_THRESHOLD = 2

random.seed(42)


# --------------------------------------------------------------------------- #
# Helpers to convert KG node IDs back to original IDs
# --------------------------------------------------------------------------- #

def strip_prefix(node, prefix):
    """Remove the type prefix from a node ID (e.g. 'item_1234' -> '1234')."""
    expected = prefix + '_'
    if not node.startswith(expected):
        raise ValueError(f"Node {node} does not start with expected prefix {expected}")
    return node[len(expected):]


# --------------------------------------------------------------------------- #
# Spot checks per relation
# --------------------------------------------------------------------------- #

def check_owns(df, kg_df, n_samples):
    """Verify that (client, owns, machine) triples correspond to real rows."""
    logger.info("\n--- Spot check: owns ---")
    sub = kg_df[kg_df['relation'] == 'owns'].sample(n=n_samples, random_state=1)

    failures = 0
    for _, row in sub.iterrows():
        client_id = strip_prefix(row['head'], 'client')
        machine_id = strip_prefix(row['tail'], 'machine')

        # Check the pair exists in the CSV
        match = df[
            (df[COL_CLIENT].astype(str) == client_id) &
            (df[COL_MACHINE].astype(str) == machine_id)
        ]
        if len(match) == 0:
            logger.error(f"  FAIL: ({client_id}, owns, {machine_id}) not in CSV")
            failures += 1

    if failures == 0:
        logger.info(f"  PASS: all {n_samples} 'owns' triples found in CSV")
    return failures


def check_instance_of(df, kg_df, n_samples):
    """Verify that (machine, instance_of, model) triples correspond to real rows."""
    logger.info("\n--- Spot check: instance_of ---")
    sub = kg_df[kg_df['relation'] == 'instance_of'].sample(n=n_samples, random_state=2)

    failures = 0
    for _, row in sub.iterrows():
        machine_id = strip_prefix(row['head'], 'machine')
        model_id = strip_prefix(row['tail'], 'model')

        match = df[
            (df[COL_MACHINE].astype(str) == machine_id) &
            (df[COL_MODEL].astype(str) == model_id)
        ]
        if len(match) == 0:
            logger.error(f"  FAIL: ({machine_id}, instance_of, {model_id}) not in CSV")
            failures += 1

    if failures == 0:
        logger.info(f"  PASS: all {n_samples} 'instance_of' triples found in CSV")
    return failures


def check_bought(df, kg_df, n_samples):
    """Verify that (client, bought, item) triples correspond to real rows."""
    logger.info("\n--- Spot check: bought ---")
    sub = kg_df[kg_df['relation'] == 'bought'].sample(n=n_samples, random_state=3)

    failures = 0
    for _, row in sub.iterrows():
        client_id = strip_prefix(row['head'], 'client')
        item_id = strip_prefix(row['tail'], 'item')

        match = df[
            (df[COL_CLIENT].astype(str) == client_id) &
            (df[COL_ITEM].astype(str) == item_id)
        ]
        if len(match) == 0:
            logger.error(f"  FAIL: ({client_id}, bought, {item_id}) not in CSV")
            failures += 1

    if failures == 0:
        logger.info(f"  PASS: all {n_samples} 'bought' triples found in CSV")
    return failures


def check_compatible_with(df, kg_df, n_samples):
    """Verify that (item, compatible_with, machine) triples correspond to real rows."""
    logger.info("\n--- Spot check: compatible_with ---")
    sub = kg_df[kg_df['relation'] == 'compatible_with'].sample(n=n_samples, random_state=4)

    failures = 0
    for _, row in sub.iterrows():
        item_id = strip_prefix(row['head'], 'item')
        machine_id = strip_prefix(row['tail'], 'machine')

        match = df[
            (df[COL_ITEM].astype(str) == item_id) &
            (df[COL_MACHINE].astype(str) == machine_id)
        ]
        if len(match) == 0:
            logger.error(f"  FAIL: ({item_id}, compatible_with, {machine_id}) not in CSV")
            failures += 1

    if failures == 0:
        logger.info(f"  PASS: all {n_samples} 'compatible_with' triples found in CSV")
    return failures


def check_compatible_with_model(df, kg_df, n_samples, threshold):
    """
    Verify (item, compatible_with_model, model) triples:
      1. The pair (item, model) appears in the CSV
      2. The pair is supported by at least `threshold` distinct machines
    """
    logger.info("\n--- Spot check: compatible_with_model ---")
    sub = kg_df[kg_df['relation'] == 'compatible_with_model'].sample(n=n_samples, random_state=5)

    failures_existence = 0
    failures_threshold = 0

    for _, row in sub.iterrows():
        item_id = strip_prefix(row['head'], 'item')
        model_id = strip_prefix(row['tail'], 'model')

        match = df[
            (df[COL_ITEM].astype(str) == item_id) &
            (df[COL_MODEL].astype(str) == model_id)
        ]
        if len(match) == 0:
            logger.error(f"  FAIL existence: ({item_id}, ..., {model_id}) not in CSV")
            failures_existence += 1
            continue

        n_distinct_machines = match[COL_MACHINE].nunique()
        if n_distinct_machines < threshold:
            logger.error(
                f"  FAIL threshold: ({item_id}, ..., {model_id}) "
                f"has only {n_distinct_machines} distinct machines (< {threshold})"
            )
            failures_threshold += 1

    if failures_existence == 0 and failures_threshold == 0:
        logger.info(
            f"  PASS: all {n_samples} 'compatible_with_model' triples are in CSV "
            f"and respect the k>={threshold} threshold"
        )
    return failures_existence + failures_threshold


# --------------------------------------------------------------------------- #
# Global invariants
# --------------------------------------------------------------------------- #

def check_no_duplicates(kg_df):
    """The KG must have no duplicate triples."""
    logger.info("\n--- Invariant: no duplicates ---")
    n_dup = kg_df.duplicated().sum()
    if n_dup > 0:
        logger.error(f"  FAIL: {n_dup:,} duplicate triples found")
        return 1
    logger.info("  PASS: no duplicates")
    return 0


def check_node_prefix_consistency(kg_df):
    """
    For each relation, the head and tail prefixes must be consistent.
    e.g. 'owns' must always go from client_* to machine_*.
    """
    logger.info("\n--- Invariant: head/tail prefix consistency per relation ---")

    expected = {
        'owns': ('client', 'machine'),
        'instance_of': ('machine', 'model'),
        'bought': ('client', 'item'),
        'compatible_with': ('item', 'machine'),
        'compatible_with_model': ('item', 'model'),
    }

    failures = 0
    for rel, (h_pref, t_pref) in expected.items():
        sub = kg_df[kg_df['relation'] == rel]
        bad_h = (~sub['head'].str.startswith(h_pref + '_')).sum()
        bad_t = (~sub['tail'].str.startswith(t_pref + '_')).sum()
        if bad_h > 0 or bad_t > 0:
            logger.error(
                f"  FAIL: relation '{rel}' has {bad_h} bad heads, {bad_t} bad tails"
            )
            failures += 1
        else:
            logger.info(f"  PASS: '{rel}' uses ({h_pref}_*, {t_pref}_*)")

    return failures


def check_machine_to_model_unique(kg_df):
    """In the final KG, each machine must have exactly one model."""
    logger.info("\n--- Invariant: each machine has exactly one model ---")
    sub = kg_df[kg_df['relation'] == 'instance_of']
    counts = sub.groupby('head')['tail'].nunique()
    bad = counts[counts > 1]
    if len(bad) > 0:
        logger.error(f"  FAIL: {len(bad):,} machines have multiple models")
        return 1
    logger.info(f"  PASS: all {len(counts):,} machines have exactly one model")
    return 0


def check_no_taska_leakage(kg_df, valid_df, test_df):
    """
    Task A holdout triples must NOT appear in the KG.
    Final guard: paranoid recheck after build.
    """
    logger.info("\n--- Invariant: no Task A leakage ---")
    kg_set = set(map(tuple, kg_df.values))

    leaks_v = sum(1 for r in valid_df.values if tuple(r) in kg_set)
    leaks_t = sum(1 for r in test_df.values if tuple(r) in kg_set)

    if leaks_v > 0 or leaks_t > 0:
        logger.error(f"  FAIL: {leaks_v} valid + {leaks_t} test triples leaked into KG")
        return 1
    logger.info("  PASS: no Task A leakage detected")
    return 0


def check_compatible_with_completeness(df, kg_df, valid_df, test_df):
    """
    The union of (compatible_with in KG) + (valid) + (test) should equal
    the set of all distinct (item, machine) pairs in the CSV.
    """
    logger.info("\n--- Invariant: compatible_with split is complete ---")

    # Pairs from CSV
    csv_pairs = set(
        df[[COL_ITEM, COL_MACHINE]]
        .dropna()
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    # Abbiamo rimosso int(i) per gestire item alfanumerici
    csv_pairs = {(f"item_{str(i)}", f"machine_{m}") for (i, m) in csv_pairs}

    # Pairs from KG
    cw_kg = kg_df[kg_df['relation'] == 'compatible_with']
    kg_pairs = set(zip(cw_kg['head'], cw_kg['tail']))

    # Pairs from valid + test
    val_pairs = set(zip(valid_df['head'], valid_df['tail']))
    test_pairs = set(zip(test_df['head'], test_df['tail']))

    union = kg_pairs | val_pairs | test_pairs
    missing = csv_pairs - union
    extra = union - csv_pairs

    if len(missing) == 0 and len(extra) == 0:
        logger.info(
            f"  PASS: compatible_with split covers all {len(csv_pairs):,} CSV pairs"
        )
        return 0

    logger.error(f"  FAIL: missing {len(missing):,}, extra {len(extra):,}")
    if missing:
        logger.error(f"    sample missing: {list(missing)[:3]}")
    if extra:
        logger.error(f"    sample extra: {list(extra)[:3]}")
    return 1


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    logger.info("=" * 70)
    logger.info("KG VALIDATION")
    logger.info("=" * 70)

    df = pd.read_csv(TRAIN_CSV, sep=';')
    kg_df = pd.read_csv(KG_PATH, sep='\t')
    valid_df = pd.read_csv(TASKA_VALID, sep='\t')
    test_df = pd.read_csv(TASKA_TEST, sep='\t')

    logger.info(f"Loaded CSV  : {len(df):,} rows")
    logger.info(f"Loaded KG   : {len(kg_df):,} triples")
    logger.info(f"Loaded valid: {len(valid_df):,} triples")
    logger.info(f"Loaded test : {len(test_df):,} triples")

    total_failures = 0

    # ---- Layer 1: spot checks ----
    logger.info("\n" + "=" * 70)
    logger.info("LAYER 1: SAMPLED SPOT CHECKS")
    logger.info("=" * 70)
    total_failures += check_owns(df, kg_df, N_SAMPLES_PER_RELATION)
    total_failures += check_instance_of(df, kg_df, N_SAMPLES_PER_RELATION)
    total_failures += check_bought(df, kg_df, N_SAMPLES_PER_RELATION)
    total_failures += check_compatible_with(df, kg_df, N_SAMPLES_PER_RELATION)
    total_failures += check_compatible_with_model(
        df, kg_df, N_SAMPLES_PER_RELATION, COMPATIBLE_WITH_MODEL_THRESHOLD
    )

    # ---- Layer 2: global invariants ----
    logger.info("\n" + "=" * 70)
    logger.info("LAYER 2: GLOBAL INVARIANTS")
    logger.info("=" * 70)
    total_failures += check_no_duplicates(kg_df)
    total_failures += check_node_prefix_consistency(kg_df)
    total_failures += check_machine_to_model_unique(kg_df)
    total_failures += check_no_taska_leakage(kg_df, valid_df, test_df)
    total_failures += check_compatible_with_completeness(df, kg_df, valid_df, test_df)

    # ---- Verdict ----
    logger.info("\n" + "=" * 70)
    if total_failures == 0:
        logger.info("ALL CHECKS PASSED")
    else:
        logger.error(f"{total_failures} CHECKS FAILED -- review the log above")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
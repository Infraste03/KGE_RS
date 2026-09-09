"""
Load the Knowledge Graph for Step 4 (alternate learning).

This module reads the three TSV files produced by build_kg.py:
  - kg_train.tsv          : training triples (used for the Task A loss)
  - taskA_valid.tsv       : validation triples for compatible_with link prediction
  - taskA_test.tsv        : test triples for compatible_with link prediction

It builds the integer-based representation needed by KGE models (PyKEEN, custom
TransE, etc.): every entity and every relation gets a unique integer ID.

Output:
    A dictionary with the following keys:
        - 'train_triples'        : np.ndarray of shape (N, 3), int64
        - 'valid_triples'        : np.ndarray of shape (M, 3), int64
        - 'test_triples'         : np.ndarray of shape (K, 3), int64
        - 'entity_to_id'         : dict[str, int]
        - 'id_to_entity'         : dict[int, str]
        - 'relation_to_id'       : dict[str, int]
        - 'id_to_relation'       : dict[int, str]
        - 'num_entities'         : int
        - 'num_relations'        : int
        - 'entity_type_indices'  : dict[str, list[int]]
            Maps each type ('item', 'machine', 'model', 'client') to the list
            of integer IDs of entities of that type. Useful for type-constrained
            negative sampling and for the shared embedding setup.

Usage:
    from load_kg import load_kg

    kg = load_kg(
        train_path='data/processed/kg_train.tsv',
        valid_path='data/processed/taskA_valid.tsv',
        test_path='data/processed/taskA_test.tsv',
    )
    print(kg['num_entities'], kg['num_relations'])
"""

import os
import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Constants: entity type prefixes used in build_kg.py
# --------------------------------------------------------------------------- #

ENTITY_TYPE_PREFIXES = ['item', 'machine', 'model', 'client']


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _read_tsv(path: str) -> pd.DataFrame:
    """Read a TSV file with the standard (head, relation, tail) schema."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected file not found: {path}")
    df = pd.read_csv(path, sep='\t')
    expected_cols = {'head', 'relation', 'tail'}
    if not expected_cols.issubset(df.columns):
        raise ValueError(
            f"File {path} is missing expected columns. "
            f"Found: {list(df.columns)}, expected at least: {expected_cols}"
        )
    return df


def _entity_type(entity_string: str) -> str:
    """Extract the type prefix from an entity string (e.g. 'item_1234' -> 'item')."""
    for prefix in ENTITY_TYPE_PREFIXES:
        if entity_string.startswith(prefix + '_'):
            return prefix
    raise ValueError(f"Entity '{entity_string}' has no recognized type prefix")


def _build_mappings(all_entities: set, all_relations: set):
    """
    Build entity-to-id and relation-to-id mappings.

    Sorting before assigning IDs makes the mapping deterministic across runs:
    same dataset -> same IDs -> same models can be loaded without surprises.
    """
    sorted_entities = sorted(all_entities)
    sorted_relations = sorted(all_relations)

    entity_to_id = {e: i for i, e in enumerate(sorted_entities)}
    id_to_entity = {i: e for e, i in entity_to_id.items()}

    relation_to_id = {r: i for i, r in enumerate(sorted_relations)}
    id_to_relation = {i: r for r, i in relation_to_id.items()}

    return entity_to_id, id_to_entity, relation_to_id, id_to_relation


def _triples_to_ids(
    df: pd.DataFrame,
    entity_to_id: dict,
    relation_to_id: dict,
    split_name: str,
) -> np.ndarray:
    """Convert a DataFrame of string triples to an int64 numpy array."""
    n = len(df)
    arr = np.empty((n, 3), dtype=np.int64)

    unknown_entities = set()
    unknown_relations = set()

    for i, row in enumerate(df.itertuples(index=False)):
        h, r, t = row.head, row.relation, row.tail

        if h not in entity_to_id:
            unknown_entities.add(h)
        if t not in entity_to_id:
            unknown_entities.add(t)
        if r not in relation_to_id:
            unknown_relations.add(r)

        # We index lookup defensively: if an unknown is found, we still build the
        # array but raise after, with a precise list.
        arr[i, 0] = entity_to_id.get(h, -1)
        arr[i, 1] = relation_to_id.get(r, -1)
        arr[i, 2] = entity_to_id.get(t, -1)

    if unknown_entities or unknown_relations:
        msg = f"In split '{split_name}': "
        if unknown_entities:
            sample = list(unknown_entities)[:5]
            msg += f"{len(unknown_entities)} unknown entities (sample: {sample}). "
        if unknown_relations:
            msg += f"unknown relations: {unknown_relations}. "
        raise ValueError(msg)

    return arr


# --------------------------------------------------------------------------- #
# Main loader
# --------------------------------------------------------------------------- #

def load_kg(
    train_path: str,
    valid_path: str,
    test_path: str,
    verbose: bool = True,
) -> dict:
    """
    Load and integer-encode the KG triples for Step 4.

    Args:
        train_path: path to kg_train.tsv
        valid_path: path to taskA_valid.tsv
        test_path:  path to taskA_test.tsv
        verbose:    if True, print summary statistics

    Returns:
        Dictionary as described in the module docstring.
    """
    if verbose:
        logger.info("=" * 70)
        logger.info("LOADING KG FOR STEP 4 (ALTERNATE LEARNING)")
        logger.info("=" * 70)

    # ---- Read TSVs ----
    train_df = _read_tsv(train_path)
    valid_df = _read_tsv(valid_path)
    test_df = _read_tsv(test_path)

    if verbose:
        logger.info(f"Loaded {len(train_df):,} training triples from {train_path}")
        logger.info(f"Loaded {len(valid_df):,} validation triples from {valid_path}")
        logger.info(f"Loaded {len(test_df):,} test triples from {test_path}")

    # ---- Collect all entities and relations across the three splits ----
    all_entities = set()
    all_relations = set()

    for df in [train_df, valid_df, test_df]:
        all_entities.update(df['head'].unique())
        all_entities.update(df['tail'].unique())
        all_relations.update(df['relation'].unique())

    # ---- Sanity: every entity should have a recognized type prefix ----
    for e in all_entities:
        _entity_type(e)  # raises ValueError if prefix is unknown

    # ---- Build ID mappings ----
    entity_to_id, id_to_entity, relation_to_id, id_to_relation = _build_mappings(
        all_entities, all_relations
    )

    # ---- Convert each split to int64 numpy arrays ----
    train_triples = _triples_to_ids(train_df, entity_to_id, relation_to_id, 'train')
    valid_triples = _triples_to_ids(valid_df, entity_to_id, relation_to_id, 'valid')
    test_triples = _triples_to_ids(test_df, entity_to_id, relation_to_id, 'test')

    # ---- Group entity IDs by type (useful for typed negative sampling) ----
    entity_type_indices = {prefix: [] for prefix in ENTITY_TYPE_PREFIXES}
    for entity_str, eid in entity_to_id.items():
        entity_type_indices[_entity_type(entity_str)].append(eid)
    # Sort each list for determinism
    for k in entity_type_indices:
        entity_type_indices[k].sort()

    # ---- Print summary ----
    if verbose:
        logger.info("\n--- Summary ---")
        logger.info(f"Total entities : {len(entity_to_id):,}")
        for prefix in ENTITY_TYPE_PREFIXES:
            logger.info(
                f"  {prefix:8s}: {len(entity_type_indices[prefix]):,}"
            )
        logger.info(f"Total relations: {len(relation_to_id):,}")
        for rel, rid in sorted(relation_to_id.items(), key=lambda x: x[1]):
            n_in_train = (train_triples[:, 1] == rid).sum()
            logger.info(f"  {rel:25s} (id={rid}): {n_in_train:,} triples in train")
        logger.info(f"Train triples  : {len(train_triples):,}")
        logger.info(f"Valid triples  : {len(valid_triples):,}")
        logger.info(f"Test triples   : {len(test_triples):,}")
        logger.info("=" * 70)

    return {
        'train_triples': train_triples,
        'valid_triples': valid_triples,
        'test_triples': test_triples,
        'entity_to_id': entity_to_id,
        'id_to_entity': id_to_entity,
        'relation_to_id': relation_to_id,
        'id_to_relation': id_to_relation,
        'num_entities': len(entity_to_id),
        'num_relations': len(relation_to_id),
        'entity_type_indices': entity_type_indices,
    }


# --------------------------------------------------------------------------- #
# Standalone smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    TRAIN_PATH = os.path.join(base_dir, "data", "processed", "kg_train.tsv")
    VALID_PATH = os.path.join(base_dir, "data", "processed", "taskA_valid.tsv")
    TEST_PATH = os.path.join(base_dir, "data", "processed", "taskA_test.tsv")

    kg = load_kg(TRAIN_PATH, VALID_PATH, TEST_PATH, verbose=True)

    # ---- Sanity assertions ----
    assert kg['num_entities'] > 0, "No entities loaded"
    assert kg['num_relations'] > 0, "No relations loaded"
    assert kg['train_triples'].dtype == np.int64
    assert kg['train_triples'].shape[1] == 3
    assert (kg['train_triples'] >= 0).all(), "Negative IDs in train_triples"
    assert (kg['train_triples'][:, 0] < kg['num_entities']).all()
    assert (kg['train_triples'][:, 2] < kg['num_entities']).all()
    assert (kg['train_triples'][:, 1] < kg['num_relations']).all()

    print("\nSmoke test PASSED")
    print(f"  num_entities  = {kg['num_entities']}")
    print(f"  num_relations = {kg['num_relations']}")
    print(f"  train_triples shape = {kg['train_triples'].shape}")
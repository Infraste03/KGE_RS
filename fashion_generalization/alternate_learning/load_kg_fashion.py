"""
load_kg_fashion.py — Load the Knowledge Graph for Fashion alternate learning.

Reads the three TSV files produced by the Fashion KG building pipeline:
  - kg_train.tsv       : training triples
  - taskA_valid.tsv    : validation triples (compatible_with link prediction)
  - taskA_test.tsv     : test triples

Entity types in the Fashion KG:
  - item     : products (item_XXXXX)
  - category : categories (category_XXXXX)
  - brand    : brands (brand_XXXXX)

Relations (v1, 3 relations):
  - compatible_with  : item -> item
  - belongs_to       : item -> category
  - belongs_to_brand : item -> brand

Usage:
    from load_kg_fashion import load_kg_fashion

    kg = load_kg_fashion(
        train_path='data/processed/kg_train.tsv',
        valid_path='data/processed/taskA_valid.tsv',
        test_path='data/processed/taskA_test.tsv',
    )
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
# Entity type prefixes for the Fashion KG
# --------------------------------------------------------------------------- #

ENTITY_TYPE_PREFIXES = ['item', 'category', 'brand']


# --------------------------------------------------------------------------- #
# Helpers (identical to B2B, only ENTITY_TYPE_PREFIXES differs)
# --------------------------------------------------------------------------- #

def _read_tsv(path: str) -> pd.DataFrame:
    """Read a TSV file with the standard (head, relation, tail) schema."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, sep='\t')
    expected_cols = {'head', 'relation', 'tail'}
    if not expected_cols.issubset(df.columns):
        raise ValueError(
            f"File {path} is missing expected columns. "
            f"Found: {list(df.columns)}, expected: {expected_cols}"
        )
    return df


def _entity_type(entity_string: str) -> str:
    """Extract the type prefix from an entity string (e.g. 'item_1234' -> 'item')."""
    for prefix in ENTITY_TYPE_PREFIXES:
        if entity_string.startswith(prefix + '_'):
            return prefix
    raise ValueError(
        f"Entity '{entity_string}' does not have a recognized prefix. "
        f"Expected one of: {ENTITY_TYPE_PREFIXES}"
    )


def _build_mappings(all_entities: set, all_relations: set):
    """
    Build entity-to-id and relation-to-id mappings.
    Alphabetical ordering guarantees determinism: same dataset -> same IDs.
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
    """Convert a DataFrame of string triples into an int64 array."""
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

        arr[i, 0] = entity_to_id.get(h, -1)
        arr[i, 1] = relation_to_id.get(r, -1)
        arr[i, 2] = entity_to_id.get(t, -1)

    if unknown_entities or unknown_relations:
        msg = f"Split '{split_name}': "
        if unknown_entities:
            sample = list(unknown_entities)[:5]
            msg += f"{len(unknown_entities)} unknown entities (sample: {sample}). "
        if unknown_relations:
            msg += f"unknown relations: {unknown_relations}."
        raise ValueError(msg)

    return arr


# --------------------------------------------------------------------------- #
# Main loader
# --------------------------------------------------------------------------- #

def load_kg_fashion(
    train_path: str,
    valid_path: str,
    test_path: str,
    verbose: bool = True,
) -> dict:
    """
    Load and numerically encode Fashion KG triples for Step 4.

    Returns:
        dict with the following keys:
          - 'train_triples'       : np.ndarray (N, 3) int64
          - 'valid_triples'       : np.ndarray (M, 3) int64
          - 'test_triples'        : np.ndarray (K, 3) int64
          - 'entity_to_id'        : dict[str, int]
          - 'id_to_entity'        : dict[int, str]
          - 'relation_to_id'      : dict[str, int]
          - 'id_to_relation'      : dict[int, str]
          - 'num_entities'        : int
          - 'num_relations'       : int
          - 'entity_type_indices' : dict[str, list[int]]
              Keys: 'item', 'category', 'brand'
    """
    if verbose:
        logger.info("=" * 70)
        logger.info("LOADING FASHION KG FOR STEP 4 (ALTERNATE LEARNING)")
        logger.info("=" * 70)

    # ---- Read TSV files ----
    train_df = _read_tsv(train_path)
    valid_df = _read_tsv(valid_path)
    test_df  = _read_tsv(test_path)

    if verbose:
        logger.info(f"Train triples: {len(train_df):,}  from {train_path}")
        logger.info(f"Valid triples: {len(valid_df):,}  from {valid_path}")
        logger.info(f"Test  triples: {len(test_df):,}   from {test_path}")

    # ---- Collect entities and relations from all three splits ----
    all_entities = set()
    all_relations = set()
    for df in [train_df, valid_df, test_df]:
        all_entities.update(df['head'].unique())
        all_entities.update(df['tail'].unique())
        all_relations.update(df['relation'].unique())

    # ---- Sanity check: every entity must have a recognized prefix ----
    for e in all_entities:
        _entity_type(e)  # raises ValueError if the prefix is not in ENTITY_TYPE_PREFIXES

    # ---- Build ID mappings ----
    entity_to_id, id_to_entity, relation_to_id, id_to_relation = _build_mappings(
        all_entities, all_relations
    )

    # ---- Convert triples to int64 ----
    train_triples = _triples_to_ids(train_df, entity_to_id, relation_to_id, 'train')
    valid_triples = _triples_to_ids(valid_df, entity_to_id, relation_to_id, 'valid')
    test_triples  = _triples_to_ids(test_df,  entity_to_id, relation_to_id, 'test')

    # ---- Group IDs by entity type ----
    entity_type_indices = {prefix: [] for prefix in ENTITY_TYPE_PREFIXES}
    for entity_str, eid in entity_to_id.items():
        entity_type_indices[_entity_type(entity_str)].append(eid)
    for k in entity_type_indices:
        entity_type_indices[k].sort()

    # ---- Print summary ----
    if verbose:
        logger.info("\n--- Summary ---")
        logger.info(f"Total entities   : {len(entity_to_id):,}")
        for prefix in ENTITY_TYPE_PREFIXES:
            logger.info(f"  {prefix:20s}: {len(entity_type_indices[prefix]):,}")
        logger.info(f"Total relations  : {len(relation_to_id):,}")
        for rel, rid in sorted(relation_to_id.items(), key=lambda x: x[1]):
            n_in_train = (train_triples[:, 1] == rid).sum()
            logger.info(f"  {rel:30s} (id={rid}): {n_in_train:,} training triples")
        logger.info(f"Train triples    : {len(train_triples):,}")
        logger.info(f"Valid triples    : {len(valid_triples):,}")
        logger.info(f"Test  triples    : {len(test_triples):,}")
        logger.info("=" * 70)

    return {
        'train_triples'      : train_triples,
        'valid_triples'      : valid_triples,
        'test_triples'       : test_triples,
        'entity_to_id'       : entity_to_id,
        'id_to_entity'       : id_to_entity,
        'relation_to_id'     : relation_to_id,
        'id_to_relation'     : id_to_relation,
        'num_entities'       : len(entity_to_id),
        'num_relations'      : len(relation_to_id),
        'entity_type_indices': entity_type_indices,
    }


# --------------------------------------------------------------------------- #
# Standalone smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":

    BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    TRAIN_PATH = os.path.join(BASE, "data", "processed", "kg_train.tsv")
    VALID_PATH = os.path.join(BASE, "data", "processed", "taskA_valid.tsv")
    TEST_PATH  = os.path.join(BASE, "data", "processed", "taskA_test.tsv")

    kg = load_kg_fashion(TRAIN_PATH, VALID_PATH, TEST_PATH, verbose=True)

    # ---- Sanity assertions ----
    print("\nRunning sanity assertions...")

    assert kg['num_entities'] > 0
    assert kg['num_relations'] > 0
    assert kg['train_triples'].dtype == np.int64
    assert kg['train_triples'].shape[1] == 3
    assert (kg['train_triples'] >= 0).all(), "Negative IDs found in train_triples"
    assert (kg['train_triples'][:, 0] < kg['num_entities']).all()
    assert (kg['train_triples'][:, 2] < kg['num_entities']).all()
    assert (kg['train_triples'][:, 1] < kg['num_relations']).all()

    # Verify that all three entity types are present
    assert len(kg['entity_type_indices']['item']) > 0, "No items found"
    assert len(kg['entity_type_indices']['category']) > 0, "No categories found"
    assert len(kg['entity_type_indices']['brand']) > 0, "No brands found"

    # Verify that type sets are disjoint and cover all entities
    total_typed = sum(len(v) for v in kg['entity_type_indices'].values())
    assert total_typed == kg['num_entities'], \
        f"entity_type_indices do not cover all entities: {total_typed} vs {kg['num_entities']}"

    # Verify that 'compatible_with' is present among the relations
    assert 'compatible_with' in kg['relation_to_id'], \
        "Relation 'compatible_with' not found — check the TSV files"

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED")
    print("=" * 70)
    print(f"  num_entities          = {kg['num_entities']:,}")
    print(f"  num_relations         = {kg['num_relations']:,}")
    print(f"  items                 = {len(kg['entity_type_indices']['item']):,}")
    print(f"  categories            = {len(kg['entity_type_indices']['category']):,}")
    print(f"  brands                = {len(kg['entity_type_indices']['brand']):,}")
    print(f"  train_triples shape   = {kg['train_triples'].shape}")
    print(f"  valid_triples shape   = {kg['valid_triples'].shape}")
    print(f"  test_triples shape    = {kg['test_triples'].shape}")
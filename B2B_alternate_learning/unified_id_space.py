"""
UnifiedIdSpace: a single, consistent ID space for the alternate learning.

The problem this module solves
==============================

In Step 4 we combine three components, each with its own ID assignment:

  1. load_kg.py builds IDs for KG entities (24,900 entities total:
     21,077 items + 3,377 machines + 227 models + 219 clients).

  2. PyKEEN, used in Step 2 to train TransE, also has its own internal
     IDs. We've verified that PyKEEN sorts entities alphabetically, just
     like load_kg.py does, so the two mappings coincide entity-by-entity.
     This means that when we load best_model.pt's entity weights, row i
     corresponds to the entity whose load_kg.py id is i.

  3. RecBole, used in Step 3 to train SASRec, has its own item IDs.
     RecBole reserves id=0 for padding, then assigns ids 1..N to items
     in the order they're encountered. RecBole has 21,134 items total,
     of which 21,077 overlap with the KG and 57 are "KG-isolated"
     (they appear in user purchase sequences but have no
     compatible_with relation in the KG).

UnifiedIdSpace builds ONE id space that covers all entities ever needed,
and provides translation tables to/from each external system.

Design choice (the "primary" mapping)
=====================================

The unified id space uses load_kg.py's IDs as primary. Concretely:

  - IDs 0..24,899 come straight from load_kg.py (items, machines, models,
    clients of the KG, in alphabetical order).
  - IDs 24,900..24,956 are NEW: assigned to the 57 KG-isolated items that
    only appear in RecBole sequences.

Total: 24,957 entities in the unified id space.

Why load_kg.py's IDs as primary, not RecBole's:

  - TransE warm-start is the riskiest piece of the architecture. If we
    misalign the entity embeddings of TransE, we destroy MRR=0.2953 and
    the alternate learning starts from broken weights. Keeping load_kg.py
    IDs as primary makes the TransE warm-start a literal copy: row i in
    best_model.pt -> row i in SharedEmbedding.

  - SASRec warm-start still needs id translation, but that translation
    is fully encapsulated in this module (see translate_recbole_to_unified
    below). The math is straightforward: take SASRec's item embedding
    matrix of shape (21135, 400), and for each unified-id i, find the
    corresponding recbole_id via our_to_recbole[i], and copy that row.

What this module gives you
==========================

After calling build_unified_id_space(), you get a UnifiedIdSpace object
with these attributes and methods:

    .num_total_entities      : int (24,957 in our case)
    .num_unified_items       : int (21,134 in our case)
    .item_unified_ids        : list of unified IDs that are items
    .kg_isolated_unified_ids : list of unified IDs that are KG-isolated items
    .entity_to_unified_id    : dict[str, int] (e.g. 'item_00000E583A' -> 219)
    .unified_id_to_entity    : dict[int, str] (inverse)
    .unified_to_recbole(i)   : returns the RecBole id for unified id i (or None)
    .recbole_to_unified(rid) : returns the unified id for RecBole id rid

----------------------------------------------------------------------------
Where to put this file:
    B2B_alternate_learning/unified_id_space.py

(Same level as load_kg.py)
----------------------------------------------------------------------------
"""

import os
import sys
import logging
import pandas as pd
ImportWarning

from warnings import filterwarnings
filterwarnings("ignore", category=FutureWarning)

# Make load_kg importable when this file is in the same folder
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from load_kg import load_kg

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =========================================================================== #
# Helpers
# =========================================================================== #

def _extract_recbole_item_ids_from_inter(*inter_paths) -> set:
    """Extract the set of unique item token strings from RecBole .inter files."""
    all_items = set()
    for path in inter_paths:
        if not os.path.exists(path):
            logger.warning(f"  Inter file not found, skipping: {path}")
            continue
        df = pd.read_csv(path, sep='\t')
        item_col = None
        for col in df.columns:
            if col.startswith('item_id'):
                item_col = col
                break
        if item_col is None:
            raise ValueError(f"No 'item_id' column found in {path}")
        all_items.update(df[item_col].astype(str).unique())
    return all_items


def _get_recbole_internal_mapping(recbole_dataset_name, recbole_data_path) -> dict:
    """
    Reconstruct RecBole's item_token -> internal_id mapping by re-creating the
    Dataset object. Returns a dict like {'[PAD]': 0, '34946AC4D0': 1, ...}.

    Requires RecBole to be installed in the active Python environment.
    """
    from recbole.config import Config
    from recbole.data import create_dataset

    config = Config(
        model='SASRec',
        dataset=recbole_dataset_name,
        config_dict={
            'data_path': recbole_data_path,
            'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
            'MAX_ITEM_LIST_LENGTH': 300,
            'embedding_size': 64,
            'show_progress': False,
            'loss_type': 'BPR', # Oppure imposta train_neg_sample_args a None se vuoi lasciare CE
            'train_neg_sample_args': None
        }
    )
    dataset = create_dataset(config)
    item_field = config['ITEM_ID_FIELD']
    return dict(dataset.field2token_id[item_field])


# =========================================================================== #
# The UnifiedIdSpace class
# =========================================================================== #

class UnifiedIdSpace:
    """
    Container for the unified id space and the translation tables to/from
    the external systems (load_kg, PyKEEN, RecBole).

    Construct via the factory function build_unified_id_space() below; do not
    construct this class directly.
    """

    def __init__(
        self,
        entity_to_unified_id: dict,
        unified_id_to_entity: dict,
        item_unified_ids: list,
        kg_isolated_unified_ids: list,
        recbole_token_to_internal: dict,
        unified_to_recbole_internal: dict,
    ):
        self.entity_to_unified_id = entity_to_unified_id
        self.unified_id_to_entity = unified_id_to_entity
        self.item_unified_ids = item_unified_ids
        self.kg_isolated_unified_ids = kg_isolated_unified_ids
        self.recbole_token_to_internal = recbole_token_to_internal
        self._unified_to_recbole = unified_to_recbole_internal
        # Inverse map (RecBole internal id -> unified id), built once
        self._recbole_to_unified = {
            rid: uid for uid, rid in unified_to_recbole_internal.items()
        }

    # ----- Convenience properties ------------------------------------------ #

    @property
    def num_total_entities(self) -> int:
        return len(self.entity_to_unified_id)

    @property
    def num_unified_items(self) -> int:
        return len(self.item_unified_ids)

    @property
    def num_kg_isolated_items(self) -> int:
        return len(self.kg_isolated_unified_ids)

    # ----- Translation methods --------------------------------------------- #

    def unified_to_recbole(self, unified_id: int):
        """
        Given a unified id, return the corresponding RecBole internal id.
        Returns None if the unified id is not an item or is not present in
        RecBole (which shouldn't happen for any item, but is a safe default).
        """
        return self._unified_to_recbole.get(unified_id, None)

    def recbole_to_unified(self, recbole_id: int):
        """
        Given a RecBole internal id, return the corresponding unified id.
        Returns None if the RecBole id is not in our mapping (e.g. padding).
        """
        return self._recbole_to_unified.get(recbole_id, None)


# =========================================================================== #
# Factory function: build the unified id space
# =========================================================================== #

def build_unified_id_space(
    kg_train_path: str,
    kg_valid_path: str,
    kg_test_path: str,
    recbole_train_inter: str,
    recbole_valid_inter: str,
    recbole_test_inter: str,
    recbole_dataset_name: str = "b2b_data",
    recbole_data_path: str = "dataset",
    verbose: bool = True,
) -> UnifiedIdSpace:
    """
    Build the unified id space by combining the KG mapping (from load_kg.py)
    with the item set from RecBole.

    Returns a UnifiedIdSpace object ready to be used by the rest of Step 4.
    """
    if verbose:
        logger.info("=" * 70)
        logger.info("BUILDING UNIFIED ID SPACE FOR STEP 4")
        logger.info("=" * 70)

    # --------------------------------------------------------------------- #
    # 1. Load KG mapping
    # --------------------------------------------------------------------- #
    kg = load_kg(
        train_path=kg_train_path,
        valid_path=kg_valid_path,
        test_path=kg_test_path,
        verbose=False,
    )

    # Start from load_kg's mapping (these are our PRIMARY ids)
    entity_to_unified_id = dict(kg['entity_to_id'])  # copy
    unified_id_to_entity = dict(kg['id_to_entity'])  # copy

    # Track which unified ids are items (from the KG side)
    kg_item_unified_ids = list(kg['entity_type_indices']['item'])

    if verbose:
        logger.info(f"\n[Step 1/4] KG mapping loaded.")
        logger.info(f"  KG entities      : {len(entity_to_unified_id):,}")
        logger.info(f"  KG items         : {len(kg_item_unified_ids):,}")

    # --------------------------------------------------------------------- #
    # 2. Extract RecBole item set from .inter files
    # --------------------------------------------------------------------- #
    recbole_items_raw = _extract_recbole_item_ids_from_inter(
        recbole_train_inter, recbole_valid_inter, recbole_test_inter,
    )
    # RecBole stores items WITHOUT the 'item_' prefix (just the raw ID string),
    # while our KG stores them WITH the prefix. Normalize for comparison.
    recbole_item_strings_with_prefix = {
        f"item_{x}" for x in recbole_items_raw
    }

    if verbose:
        logger.info(f"\n[Step 2/4] RecBole item set extracted from .inter files.")
        logger.info(f"  RecBole items    : {len(recbole_items_raw):,}")

    # --------------------------------------------------------------------- #
    # 3. Identify KG-isolated items and assign new unified ids to them
    # --------------------------------------------------------------------- #
    kg_isolated = recbole_item_strings_with_prefix - set(entity_to_unified_id.keys())

    if verbose:
        logger.info(f"\n[Step 3/4] Identifying KG-isolated items ...")
        logger.info(f"  Items in BOTH (KG + RecBole): "
                    f"{len(recbole_item_strings_with_prefix & set(entity_to_unified_id.keys())):,}")
        logger.info(f"  Items in KG only            : "
                    f"{len(set(e for e in entity_to_unified_id if e.startswith('item_')) - recbole_item_strings_with_prefix):,}")
        logger.info(f"  Items in RecBole only       : {len(kg_isolated):,}")

    # Assign new unified ids to KG-isolated items, starting from the next
    # available id (right after all KG entities). Sort for determinism.
    next_id = max(entity_to_unified_id.values()) + 1
    kg_isolated_sorted = sorted(kg_isolated)
    kg_isolated_unified_ids = []

    for entity_str in kg_isolated_sorted:
        entity_to_unified_id[entity_str] = next_id
        unified_id_to_entity[next_id] = entity_str
        kg_isolated_unified_ids.append(next_id)
        next_id += 1

    # All items in the unified space (KG items + KG-isolated items)
    all_item_unified_ids = sorted(kg_item_unified_ids + kg_isolated_unified_ids)

    if verbose:
        logger.info(f"  Assigned unified ids {min(kg_isolated_unified_ids) if kg_isolated_unified_ids else 'N/A'} "
                    f"to {max(kg_isolated_unified_ids) if kg_isolated_unified_ids else 'N/A'} "
                    f"to KG-isolated items.")
        logger.info(f"  Total entities in unified id space: {len(entity_to_unified_id):,}")
        logger.info(f"  Total items in unified id space   : {len(all_item_unified_ids):,}")

    # --------------------------------------------------------------------- #
    # 4. Build the translation table to/from RecBole's internal IDs
    # --------------------------------------------------------------------- #
    if verbose:
        logger.info(f"\n[Step 4/4] Building translation table to/from RecBole ...")

    recbole_token_to_internal = _get_recbole_internal_mapping(
        recbole_dataset_name, recbole_data_path
    )

    # For each unified id that is an item, find its RecBole internal id
    unified_to_recbole_internal = {}
    items_not_in_recbole = []

    for unified_id in all_item_unified_ids:
        entity_str = unified_id_to_entity[unified_id]
        # Strip the 'item_' prefix to get the raw RecBole token
        recbole_token = entity_str[len('item_'):]
        if recbole_token in recbole_token_to_internal:
            unified_to_recbole_internal[unified_id] = recbole_token_to_internal[recbole_token]
        else:
            items_not_in_recbole.append((unified_id, entity_str))

    if verbose:
        logger.info(f"  Unified items mapped to RecBole: {len(unified_to_recbole_internal):,}")
        logger.info(f"  Unified items NOT in RecBole   : {len(items_not_in_recbole):,}")
        if items_not_in_recbole:
            logger.info(f"    These items exist in the KG but not in RecBole purchase sequences.")
            logger.info(f"    They will be trained by Task A only (via compatible_with).")
            logger.info(f"    Sample: {items_not_in_recbole[:3]}")

    # --------------------------------------------------------------------- #
    # 5. Final summary
    # --------------------------------------------------------------------- #
    if verbose:
        logger.info("\n" + "=" * 70)
        logger.info("UNIFIED ID SPACE BUILT")
        logger.info("=" * 70)
        logger.info(f"  Total entities                    : {len(entity_to_unified_id):,}")
        logger.info(f"  Total items (any type)            : {len(all_item_unified_ids):,}")
        logger.info(f"    - Items shared (KG + RecBole)   : {len(unified_to_recbole_internal):,}")
        logger.info(f"    - Items only in KG              : {len(items_not_in_recbole):,}")
        logger.info(f"    - Items only in RecBole         : {len(kg_isolated_unified_ids):,}")
        logger.info(f"  Other KG entities                 : "
                    f"{len(entity_to_unified_id) - len(all_item_unified_ids):,}")

    return UnifiedIdSpace(
        entity_to_unified_id=entity_to_unified_id,
        unified_id_to_entity=unified_id_to_entity,
        item_unified_ids=all_item_unified_ids,
        kg_isolated_unified_ids=kg_isolated_unified_ids,
        recbole_token_to_internal=recbole_token_to_internal,
        unified_to_recbole_internal=unified_to_recbole_internal,
    )


# =========================================================================== #
# Standalone smoke test
# =========================================================================== #

if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    KG_TRAIN_PATH = os.path.join(base_dir, "data", "processed", "kg_train.tsv")
    KG_VALID_PATH = os.path.join(base_dir, "data", "processed", "taskA_valid.tsv")
    KG_TEST_PATH = os.path.join(base_dir, "data", "processed", "taskA_test.tsv")

    RECBOLE_TRAIN_INTER = os.path.join(base_dir, "dataset", "b2b_data", "b2b_data.train.inter")
    RECBOLE_VALID_INTER = os.path.join(base_dir, "dataset", "b2b_data", "b2b_data.valid.inter")
    RECBOLE_TEST_INTER = os.path.join(base_dir, "dataset", "b2b_data", "b2b_data.test.inter")

    # Build the unified id space
    space = build_unified_id_space(
        kg_train_path=KG_TRAIN_PATH,
        kg_valid_path=KG_VALID_PATH,
        kg_test_path=KG_TEST_PATH,
        recbole_train_inter=RECBOLE_TRAIN_INTER,
        recbole_valid_inter=RECBOLE_VALID_INTER,
        recbole_test_inter=RECBOLE_TEST_INTER,
        recbole_dataset_name="b2b_data",
        recbole_data_path=os.path.join(base_dir, "dataset"),
        verbose=True,
    )

    # ---- Sanity assertions ----
    print("\nRunning sanity assertions on the unified id space...")

    # 1. Size sanity
    assert space.num_total_entities == 24_957, \
        f"Expected 24957 total entities, got {space.num_total_entities}"
    assert space.num_unified_items == 21_134, \
        f"Expected 21134 items, got {space.num_unified_items}"
    assert space.num_kg_isolated_items == 57, \
        f"Expected 57 KG-isolated items, got {space.num_kg_isolated_items}"

    # 2. Translation sanity: pick a known item and verify the round-trip works
    test_item = 'item_00000E583A'
    if test_item in space.entity_to_unified_id:
        uid = space.entity_to_unified_id[test_item]
        rid = space.unified_to_recbole(uid)
        if rid is not None:
            uid_back = space.recbole_to_unified(rid)
            assert uid == uid_back, (
                f"Round-trip failed: {test_item} -> {uid} -> recbole {rid} -> {uid_back}"
            )
            print(f"  Round-trip test: '{test_item}' -> uid={uid} -> recbole={rid} -> uid={uid_back}")

    # 3. KG-isolated items should NOT be in the KG mapping but should be in RecBole
    if space.kg_isolated_unified_ids:
        sample_isolated_uid = space.kg_isolated_unified_ids[0]
        sample_isolated_str = space.unified_id_to_entity[sample_isolated_uid]
        rid = space.unified_to_recbole(sample_isolated_uid)
        assert rid is not None, (
            f"KG-isolated item {sample_isolated_str} (uid={sample_isolated_uid}) "
            f"should have a RecBole id"
        )
        print(f"  KG-isolated item: '{sample_isolated_str}' -> uid={sample_isolated_uid}, "
              f"recbole={rid}")

    # 4. Padding check: RecBole's padding token (id=0) should NOT translate to any unified id
    pad_uid = space.recbole_to_unified(0)
    assert pad_uid is None, f"RecBole padding (id=0) should not map to any unified id, got {pad_uid}"
    print(f"  Padding check: RecBole id=0 -> unified id=None (correct)")

    print("\n" + "=" * 70)
    print("UNIFIED ID SPACE SMOKE TEST PASSED")
    print("=" * 70)
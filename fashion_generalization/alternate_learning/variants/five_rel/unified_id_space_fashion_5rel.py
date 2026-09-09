import os
import sys
import logging
import pandas as pd
from warnings import filterwarnings

filterwarnings("ignore", category=FutureWarning)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from load_kg_fashion_5rel import load_kg_fashion


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def _extract_item_tokens_from_inter(inter_path: str) -> set:

    if not os.path.exists(inter_path):
        raise FileNotFoundError(
            f".inter file not found: {inter_path}"
        )

    df = pd.read_csv(inter_path, sep='\t')

    item_col = None

    for col in df.columns:
        if col.startswith('item_id'):
            item_col = col
            break

    if item_col is None:
        raise ValueError(
            f"No 'item_id' column found in {inter_path}. "
            f"Columns: {list(df.columns)}"
        )

    tokens = set(
        df[item_col].astype(str).unique()
    )

    logger.info(
        f"  Extracted {len(tokens):,} unique item tokens "
        f"from {inter_path}"
    )

    return tokens


def _get_recbole_internal_mapping(data_path: str) -> dict:

    from recbole.config import Config
    from recbole.data import create_dataset

    config = Config(
        model='SASRec',
        dataset='fashion',
        config_dict={
            'data_path':       data_path,
            'USER_ID_FIELD':   'user_id',
            'ITEM_ID_FIELD':   'item_id',
            'TIME_FIELD':      'timestamp',
            'load_col':        {
                'inter': [
                    'user_id',
                    'item_id',
                    'timestamp'
                ]
            },
            'field_separator': "\t",
            'eval_args': {
                'split':    {'LS': 'valid_and_test'},
                'order':    'TO',
                'group_by': 'user',
                'mode':     {
                    'valid': 'full',
                    'test': 'full'
                },
            },
            'MAX_ITEM_LIST_LENGTH': 50,
            'embedding_size':       64,
            'show_progress':        False,
            'train_neg_sample_args': None,
        }
    )

    dataset = create_dataset(config)

    item_field = config['ITEM_ID_FIELD']

    mapping = dict(
        dataset.field2token_id[item_field]
    )

    logger.info(
        f"  Reconstructed RecBole mapping: "
        f"{len(mapping):,} entries "
        f"(including [PAD]=0)"
    )

    return mapping


class UnifiedIdSpace:

    def __init__(
        self,
        entity_to_unified_id: dict,
        unified_id_to_entity: dict,
        item_unified_ids: list,
        kg_isolated_unified_ids: list,
        recbole_token_to_internal: dict,
        unified_to_recbole_internal: dict,
    ):
        self.entity_to_unified_id      = entity_to_unified_id
        self.unified_id_to_entity      = unified_id_to_entity
        self.item_unified_ids          = item_unified_ids
        self.kg_isolated_unified_ids   = kg_isolated_unified_ids
        self.recbole_token_to_internal = recbole_token_to_internal
        self._unified_to_recbole       = unified_to_recbole_internal

        self._recbole_to_unified = {
            rid: uid
            for uid, rid in unified_to_recbole_internal.items()
        }

    @property
    def num_total_entities(self) -> int:
        return len(
            self.entity_to_unified_id
        )

    @property
    def num_unified_items(self) -> int:
        return len(
            self.item_unified_ids
        )

    @property
    def num_kg_isolated_items(self) -> int:
        return len(
            self.kg_isolated_unified_ids
        )

    def unified_to_recbole(
        self,
        unified_id: int
    ):
        return self._unified_to_recbole.get(
            unified_id,
            None
        )

    def recbole_to_unified(
        self,
        recbole_id: int
    ):
        return self._recbole_to_unified.get(
            recbole_id,
            None
        )


def build_unified_id_space_fashion(
    kg_train_path: str,
    kg_valid_path: str,
    kg_test_path: str,
    recbole_inter_path: str,
    recbole_data_path: str,
    verbose: bool = True,
) -> UnifiedIdSpace:

    if verbose:
        logger.info("=" * 70)
        logger.info(
            "BUILDING UNIFIED ID SPACE — FASHION 5-RELATION KG"
        )
        logger.info("=" * 70)

    kg = load_kg_fashion(
        train_path=kg_train_path,
        valid_path=kg_valid_path,
        test_path=kg_test_path,
        verbose=False,
    )

    entity_to_unified_id = dict(
        kg['entity_to_id']
    )

    unified_id_to_entity = dict(
        kg['id_to_entity']
    )

    kg_item_unified_ids = list(
        kg['entity_type_indices']['item']
    )

    if verbose:
        logger.info(
            f"  Total KG entities : "
            f"{len(entity_to_unified_id):,}"
        )

        logger.info(
            f"  Items in KG       : "
            f"{len(kg_item_unified_ids):,}"
        )

        logger.info(
            f"  Categories        : "
            f"{len(kg['entity_type_indices']['category']):,}"
        )

        logger.info(
            f"  Brands            : "
            f"{len(kg['entity_type_indices']['brand']):,}"
        )

        logger.info(
            f"  Price tiers       : "
            f"{len(kg['entity_type_indices']['pricetier']):,}"
        )

        logger.info(
            f"  Popularity tiers  : "
            f"{len(kg['entity_type_indices']['poptier']):,}"
        )


    if verbose:
        logger.info(
            "\n[Step 2/4] Extracting items from RecBole..."
        )

    recbole_tokens_raw = _extract_item_tokens_from_inter(
        recbole_inter_path
    )

    # RecBole uses tokens such as 'B005OT7MBM', while the KG uses
    # 'item_B005OT7MBM'. Add the 'item_' prefix before matching.
    recbole_items_with_prefix = {
        f"item_{t}"
        for t in recbole_tokens_raw
    }

    if verbose:
        logger.info(
            f"  Unique items in RecBole: "
            f"{len(recbole_tokens_raw):,}"
        )


    # ------------------------------------------------------------------ #
    # 3. Identify KG-isolated items and assign new IDs
    # ------------------------------------------------------------------ #

    if verbose:
        logger.info(
            "\n[Step 3/4] Identifying KG-isolated items..."
        )

    kg_entity_set = set(
        entity_to_unified_id.keys()
    )

    kg_isolated = (
        recbole_items_with_prefix
        - kg_entity_set
    )

    in_both = (
        recbole_items_with_prefix
        & kg_entity_set
    )

    only_in_kg = (
        {
            e
            for e in kg_entity_set
            if e.startswith('item_')
        }
        - recbole_items_with_prefix
    )

    if verbose:
        logger.info(
            f"  Items in BOTH KG and RecBole : "
            f"{len(in_both):,}"
        )

        logger.info(
            f"  Items ONLY in KG             : "
            f"{len(only_in_kg):,}"
        )

        logger.info(
            f"  Items ONLY in RecBole "
            f"(isolated): {len(kg_isolated):,}"
        )


    # Assign new IDs to isolated items.
    # Sorting guarantees deterministic assignment.
    next_id = (
        max(
            entity_to_unified_id.values()
        )
        + 1
    )

    kg_isolated_unified_ids = []

    for entity_str in sorted(
        kg_isolated
    ):
        entity_to_unified_id[
            entity_str
        ] = next_id

        unified_id_to_entity[
            next_id
        ] = entity_str

        kg_isolated_unified_ids.append(
            next_id
        )

        next_id += 1


    all_item_unified_ids = sorted(
        kg_item_unified_ids
        + kg_isolated_unified_ids
    )


    if verbose:

        if kg_isolated_unified_ids:
            logger.info(
                f"  IDs assigned to isolated items: "
                f"{min(kg_isolated_unified_ids)}"
                f".."
                f"{max(kg_isolated_unified_ids)}"
            )

        logger.info(
            f"  Total entities in unified space: "
            f"{len(entity_to_unified_id):,}"
        )

        logger.info(
            f"  Total items in unified space   : "
            f"{len(all_item_unified_ids):,}"
        )


    # ------------------------------------------------------------------ #
    # 4. Build unified <-> RecBole internal translation table
    # ------------------------------------------------------------------ #

    if verbose:
        logger.info(
            "\n[Step 4/4] Building RecBole translation table..."
        )


    recbole_token_to_internal = _get_recbole_internal_mapping(
        recbole_data_path
    )


    unified_to_recbole_internal = {}
    items_not_in_recbole = []


    for unified_id in all_item_unified_ids:

        entity_str = unified_id_to_entity[
            unified_id
        ]

        # Remove the 'item_' prefix.
        recbole_token = entity_str[
            len('item_'):
        ]

        if recbole_token in recbole_token_to_internal:

            unified_to_recbole_internal[
                unified_id
            ] = recbole_token_to_internal[
                recbole_token
            ]

        else:

            items_not_in_recbole.append(
                (
                    unified_id,
                    entity_str
                )
            )


    if verbose:

        logger.info(
            f"  Unified items mapped to RecBole : "
            f"{len(unified_to_recbole_internal):,}"
        )

        logger.info(
            f"  Unified items NOT in RecBole    : "
            f"{len(items_not_in_recbole):,}"
        )

        if items_not_in_recbole:

            logger.info(
                "    These items occur only in the KG "
                "and are therefore trained only by Task A."
            )

            logger.info(
                f"    Sample: "
                f"{items_not_in_recbole[:3]}"
            )


    # ------------------------------------------------------------------ #
    # 5. Final summary
    # ------------------------------------------------------------------ #

    if verbose:

        logger.info(
            "\n" + "=" * 70
        )

        logger.info(
            "UNIFIED ID SPACE BUILT"
        )

        logger.info(
            "=" * 70
        )

        logger.info(
            f"  Total entities                    : "
            f"{len(entity_to_unified_id):,}"
        )

        logger.info(
            f"  Total items                       : "
            f"{len(all_item_unified_ids):,}"
        )

        logger.info(
            f"    - Shared items (KG + RecBole)   : "
            f"{len(unified_to_recbole_internal):,}"
        )

        logger.info(
            f"    - Items only in KG              : "
            f"{len(items_not_in_recbole):,}"
        )

        logger.info(
            f"    - Items only in RecBole         : "
            f"{len(kg_isolated_unified_ids):,}"
        )

        logger.info(
            f"  Non-item entities "
            f"(category + brand + price/pop tiers): "
            f"{len(entity_to_unified_id) - len(all_item_unified_ids):,}"
        )


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

    # This file is located in:
    #   fashion_generalization/alternate_learning/variants/five_rel/
    #
    # Move three levels up to reach:
    #   fashion_generalization/
    BASE = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            ".."
        )
    )


    # Complete 5-relation KG
    KG_TRAIN = os.path.join(
        BASE,
        "data",
        "processed_v2",
        "kg_train.tsv"
    )

    KG_VALID = os.path.join(
        BASE,
        "data",
        "processed_v2",
        "taskA_valid.tsv"
    )

    KG_TEST = os.path.join(
        BASE,
        "data",
        "processed_v2",
        "taskA_test.tsv"
    )


    # Single .inter file.
    # Leave-One-Out splitting is performed internally by RecBole.
    INTER_PATH = os.path.join(
        BASE,
        "data",
        "recbole",
        "fashion",
        "fashion.inter"
    )


    # Directory containing the 'fashion/' RecBole dataset folder.
    RECBOLE_DATA = os.path.join(
        BASE,
        "data",
        "recbole"
    )


    space = build_unified_id_space_fashion(
        kg_train_path=KG_TRAIN,
        kg_valid_path=KG_VALID,
        kg_test_path=KG_TEST,
        recbole_inter_path=INTER_PATH,
        recbole_data_path=RECBOLE_DATA,
        verbose=True,
    )


    # ---- Sanity assertions ----

    print(
        "\nRunning sanity assertions..."
    )


    # 1. Internal dimensional consistency
    assert space.num_total_entities > 0
    assert space.num_unified_items > 0

    assert (
        space.num_unified_items
        <= space.num_total_entities
    )


    # 2. Round trip:
    #    unified -> RecBole -> unified must return the same ID
    sample_uid = next(
        uid
        for uid in space.item_unified_ids
        if space.unified_to_recbole(uid) is not None
    )


    rid = space.unified_to_recbole(
        sample_uid
    )

    uid_back = space.recbole_to_unified(
        rid
    )


    assert sample_uid == uid_back, \
        (
            f"Round-trip failed: "
            f"uid={sample_uid} -> "
            f"rid={rid} -> "
            f"uid={uid_back}"
        )


    sample_str = space.unified_id_to_entity[
        sample_uid
    ]


    print(
        f"  Round-trip OK: "
        f"'{sample_str}' -> "
        f"uid={sample_uid} -> "
        f"recbole={rid} -> "
        f"uid={uid_back}"
    )


    # 3. RecBole padding (id=0) must not map to a unified ID
    pad_uid = space.recbole_to_unified(
        0
    )


    assert pad_uid is None, \
        (
            f"RecBole padding (id=0) must not map "
            f"to a unified ID, got {pad_uid}"
        )


    print(
        "  Padding check: "
        "RecBole id=0 -> None (correct)"
    )


    # 4. KG-isolated items must have a valid RecBole internal ID
    if space.kg_isolated_unified_ids:

        iso_uid = space.kg_isolated_unified_ids[
            0
        ]

        iso_rid = space.unified_to_recbole(
            iso_uid
        )


        assert iso_rid is not None, \
            (
                f"KG-isolated item uid={iso_uid} "
                f"should have a RecBole ID"
            )


        iso_str = space.unified_id_to_entity[
            iso_uid
        ]


        print(
            f"  KG-isolated: "
            f"'{iso_str}' -> "
            f"uid={iso_uid}, "
            f"recbole={iso_rid}"
        )


    # 5. No duplicated IDs in item_unified_ids
    assert (
        len(space.item_unified_ids)
        == len(set(space.item_unified_ids))
    ), "Duplicate IDs found in item_unified_ids"


    print(
        "  No duplicate IDs in item_unified_ids: OK"
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "SMOKE TEST PASSED"
    )

    print(
        "=" * 70
    )

    print(
        f"  num_total_entities   = "
        f"{space.num_total_entities:,}"
    )

    print(
        f"  num_unified_items    = "
        f"{space.num_unified_items:,}"
    )

    print(
        f"  num_kg_isolated      = "
        f"{space.num_kg_isolated_items:,}"
    )
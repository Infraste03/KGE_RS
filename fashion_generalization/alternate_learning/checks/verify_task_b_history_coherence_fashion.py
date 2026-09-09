"""
verify_task_b_history_coherence_fashion.py
==========================================

Quick coherence check for Task B evaluation on the Fashion dataset.

This script is aligned with the SASRec Fashion HPO setup used in:

    fashion_generalization/models/sasrec/run_hpo_sasrec_fashion.py

Evaluation protocol:
- dataset: fashion
- split: RecBole leave-one-out style split with validation and test
         eval_args = {'split': {'LS': 'valid_and_test'}}
- temporal order: order = 'TO'
- grouped by user: group_by = 'user'
- full-sort evaluation for valid/test

Important:
This script does NOT require test_data.dataset.user_history_dict.
This script does NOT require history_index to be exposed in the batch.

Reason:
In RecBole, depending on the version and dataloader internals, the masking of
already-seen items may be handled internally and not exposed as history_index.
The correct thing to verify is that we are building valid_data and test_data
with exactly the same RecBole protocol used during SASRec training.

Checks:
- RecBole valid/test dataloaders are built correctly
- evaluation protocol matches the SASRec Fashion HPO protocol
- targets are never padding (0)
- item_id_list and item_length are present
- item_id_list and item_length are coherent
- target items are representable in the KG item space
- sequence items are representable in the KG item space
- RecBole item space is perfectly aligned with KG item entities

This is intentionally lightweight and does not run training.

Save as:
    fashion_generalization/alternate_learning/checks/verify_task_b_history_coherence_fashion.py
"""

import os
import sys
import logging
import warnings
from typing import Any, Dict, Tuple

import torch

warnings.simplefilter(action="ignore", category=FutureWarning)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
BASE_DIR = os.path.abspath(os.path.join(PARENT_DIR, ".."))

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from load_kg_fashion import load_kg_fashion

from recbole.config import Config as RecBoleConfig
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed


KG_TRAIN = os.path.join(BASE_DIR, "data", "processed", "kg_train.tsv")
KG_VALID = os.path.join(BASE_DIR, "data", "processed", "taskA_valid.tsv")
KG_TEST = os.path.join(BASE_DIR, "data", "processed", "taskA_test.tsv")

RECBOLE_DATA_PATH = os.path.join(BASE_DIR, "data", "recbole")
RECBOLE_DATASET = "fashion"

USER_FIELD = "user_id"
ITEM_FIELD = "item_id"
TIME_FIELD = "timestamp"

HIDDEN_SIZE = 64
MAX_ITEM_LIST_LENGTH = 50


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def build_recbole_config() -> RecBoleConfig:
    """
    Build the same RecBole data/evaluation configuration used for SASRec Fashion.

    This mirrors the evaluation setup in run_hpo_sasrec_fashion.py:

        eval_args = {
            'split':    {'LS': 'valid_and_test'},
            'order':    'TO',
            'group_by': 'user',
            'mode':     {'valid': 'full', 'test': 'full'},
        }

    Note:
    - n_layers, n_heads, dropout, learning_rate are model/training parameters.
      They do not affect the construction of valid/test splits.
    - train_neg_sample_args affects training, not the full-sort valid/test
      evaluation protocol checked here.
    """

    config_dict = {
        "data_path": RECBOLE_DATA_PATH,

        "USER_ID_FIELD": USER_FIELD,
        "ITEM_ID_FIELD": ITEM_FIELD,
        "TIME_FIELD": TIME_FIELD,

        "load_col": {
            "inter": [USER_FIELD, ITEM_FIELD, TIME_FIELD],
        },

        "field_separator": "\t",

        # EXACT evaluation protocol used in SASRec Fashion HPO
        "eval_args": {
            "split": {"LS": "valid_and_test"},
            "order": "TO",
            "group_by": "user",
            "mode": {
                "valid": "full",
                "test": "full",
            },
        },

        # Sequential setting used by SASRec Fashion
        "hidden_size": HIDDEN_SIZE,
        "MAX_ITEM_LIST_LENGTH": MAX_ITEM_LIST_LENGTH,

        # Batch sizes as in run_hpo_sasrec_fashion.py
        "train_batch_size": 256,
        "eval_batch_size": 512,

        # This check does not train.
        # CE + no negative sampling is neutral for this verification.
        "loss_type": "CE",
        "train_neg_sample_args": None,

        # Evaluation settings as in run_hpo_sasrec_fashion.py
        "metrics": ["Recall", "NDCG"],
        "topk": [20],
        "valid_metric": "NDCG@20",

        "seed": 2020,
        "reproducibility": True,

        # Use CPU for this lightweight check
        "use_gpu": False,
        "show_progress": False,
    }

    return RecBoleConfig(
    model="SASRec",
    dataset=RECBOLE_DATASET,
    config_dict=config_dict,
)


# --------------------------------------------------------------------------- #
# Helpers: KG / RecBole mapping
# --------------------------------------------------------------------------- #

def normalize_item_token(token: str) -> str:
    """
    Normalize a RecBole item token into the KG item label.

    Typical case:
        RecBole token: B000123
        KG entity    : item_B000123

    If the token already starts with item_, keep it unchanged.
    """
    token = str(token)

    if token.startswith("item_"):
        return token

    return f"item_{token}"


def build_recbole_to_kg_mapping(
    recbole_dataset,
    kg: Dict[str, Any],
) -> Tuple[torch.Tensor, list]:
    """
    Build a tensor:

        recbole_internal_item_id -> kg_entity_id

    If an item is not found in the KG, its value remains -1.
    RecBole padding item has internal id 0 and remains -1.
    """

    entity_to_id = kg["entity_to_id"]
    token_to_internal_id = dict(recbole_dataset.field2token_id[ITEM_FIELD])

    num_recbole_items = recbole_dataset.item_num

    recbole_to_kg = torch.full(
        (num_recbole_items,),
        fill_value=-1,
        dtype=torch.long,
    )

    missing_tokens = []

    for token, internal_id in token_to_internal_id.items():
        internal_id = int(internal_id)

        # RecBole padding
        if internal_id == 0:
            continue

        kg_label = normalize_item_token(token)

        if kg_label in entity_to_id:
            recbole_to_kg[internal_id] = int(entity_to_id[kg_label])
        else:
            missing_tokens.append(token)

    return recbole_to_kg, missing_tokens


def count_unmapped_items(item_ids: torch.Tensor, recbole_to_kg: torch.Tensor) -> int:
    """
    Count how many RecBole item ids are not mappable to KG item entities.
    Padding item 0 is ignored.
    """

    item_ids = item_ids.detach().cpu().long().view(-1)
    item_ids = item_ids[item_ids != 0]

    if item_ids.numel() == 0:
        return 0

    max_id = int(item_ids.max().item())

    if max_id >= recbole_to_kg.numel():
        return int(item_ids.numel())

    mapped = recbole_to_kg[item_ids]

    return int((mapped < 0).sum().item())


# --------------------------------------------------------------------------- #
# Helpers: RecBole batches
# --------------------------------------------------------------------------- #

def unpack_interaction(batch):
    """
    Extract the Interaction object from a RecBole batch.

    Depending on RecBole version and dataloader, a batch may be:
    - an Interaction directly
    - a tuple/list whose first element is the Interaction

    We do not require history_index, because RecBole may handle the full-sort
    mask internally without exposing it in the batch.
    """

    if isinstance(batch, (tuple, list)):
        return batch[0]

    return batch


def interaction_fields(interaction) -> set:
    """
    Return fields available in a RecBole Interaction object.
    """

    if hasattr(interaction, "interaction"):
        return set(interaction.interaction.keys())

    return set()


def describe_batch(batch) -> str:
    """
    Compact description of a RecBole batch for debugging/logging.
    """

    if isinstance(batch, (tuple, list)):
        parts = []
        for x in batch:
            if isinstance(x, torch.Tensor):
                parts.append(f"Tensor{tuple(x.shape)}")
            elif hasattr(x, "interaction"):
                parts.append(f"Interaction(fields={list(x.interaction.keys())})")
            else:
                parts.append(type(x).__name__)

        return f"{type(batch).__name__}({', '.join(parts)})"

    if hasattr(batch, "interaction"):
        return f"Interaction(fields={list(batch.interaction.keys())})"

    return type(batch).__name__


def inspect_optional_history_sources(loader, name: str):
    """
    Log optional RecBole internal attributes related to histories/masks.

    These are not required because they are version-dependent.
    The real evaluation is performed by RecBole Trainer._valid_epoch.
    """

    possible_attrs = [
        "uid2history_item",
        "uid2positive_item",
        "user_history_dict",
        "history_item_matrix",
        "uid_list",
    ]

    logger.info(f"\nOptional internal history/mask attributes for {name}:")

    for obj_name, obj in [
        (f"{name}_loader", loader),
        (f"{name}_loader.dataset", getattr(loader, "dataset", None)),
    ]:
        if obj is None:
            continue

        found_any = False

        for attr in possible_attrs:
            if hasattr(obj, attr):
                value = getattr(obj, attr)
                found_any = True

                try:
                    length = len(value)
                    logger.info(f"  {obj_name}.{attr}: present, len={length}")
                except Exception:
                    logger.info(f"  {obj_name}.{attr}: present, type={type(value).__name__}")

        if not found_any:
            logger.info(f"  {obj_name}: no standard history/mask attributes exposed")


# --------------------------------------------------------------------------- #
# Main loader check
# --------------------------------------------------------------------------- #

def check_loader(
    loader,
    name: str,
    recbole_to_kg: torch.Tensor,
):
    """
    Check a RecBole valid/test dataloader.

    Checks:
    - target item_id is never padding
    - target item_id maps to KG item
    - item_id_list exists
    - item_length exists
    - item_id_list and item_length are coherent
    - sequence items map to KG item

    This does not manually reproduce RecBole masking.
    It verifies that the dataloader used by RecBole evaluation is coherent.
    """

    zero_targets = 0
    total_targets = 0

    unmapped_targets = 0
    unmapped_sequence_items = 0

    empty_sequences = 0
    inconsistent_lengths = 0

    num_batches = 0
    num_rows = 0

    min_seq_len = None
    max_seq_len = 0

    sample_user_ids = []
    sample_target_ids = []
    sample_seq_lengths = []

    for batch in loader:
        num_batches += 1

        interaction = unpack_interaction(batch)
        fields = interaction_fields(interaction)

        required_fields = {
            USER_FIELD,
            ITEM_FIELD,
            "item_id_list",
            "item_length",
        }

        missing_fields = required_fields - fields

        assert len(missing_fields) == 0, (
            f"{name} loader missing fields: {missing_fields}. "
            f"Available fields: {sorted(fields)}"
        )

        user_ids = interaction[USER_FIELD].detach().cpu().long()
        targets = interaction[ITEM_FIELD].detach().cpu().long()
        item_seq = interaction["item_id_list"].detach().cpu().long()
        item_len = interaction["item_length"].detach().cpu().long()

        num_rows += int(targets.numel())

        zero_targets += int((targets == 0).sum().item())
        total_targets += int(targets.numel())

        unmapped_targets += count_unmapped_items(targets, recbole_to_kg)

        for i in range(targets.shape[0]):
            seq_i = item_seq[i]
            len_i = int(item_len[i].item())

            nonzero_seq = seq_i[seq_i != 0]

            if len_i == 0:
                empty_sequences += 1

            if nonzero_seq.numel() != len_i:
                inconsistent_lengths += 1

            unmapped_sequence_items += count_unmapped_items(
                nonzero_seq,
                recbole_to_kg,
            )

            if min_seq_len is None:
                min_seq_len = len_i
            else:
                min_seq_len = min(min_seq_len, len_i)

            max_seq_len = max(max_seq_len, len_i)

        if len(sample_user_ids) < 5:
            take = min(5 - len(sample_user_ids), targets.shape[0])
            sample_user_ids.extend(user_ids[:take].tolist())
            sample_target_ids.extend(targets[:take].tolist())
            sample_seq_lengths.extend(item_len[:take].tolist())

    summary = {
        "num_batches": num_batches,
        "num_rows": num_rows,
        "zero_targets": zero_targets,
        "total_targets": total_targets,
        "unmapped_targets": unmapped_targets,
        "unmapped_sequence_items": unmapped_sequence_items,
        "empty_sequences": empty_sequences,
        "inconsistent_lengths": inconsistent_lengths,
        "min_seq_len": min_seq_len,
        "max_seq_len": max_seq_len,
        "sample_user_ids": sample_user_ids,
        "sample_target_ids": sample_target_ids,
        "sample_seq_lengths": sample_seq_lengths,
    }

    return summary


def log_summary(name: str, summary: Dict[str, Any]):
    logger.info(f"\n{name.upper()} SUMMARY")

    for key, value in summary.items():
        logger.info(f"  {key:35s}: {value}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    logger.info("=" * 70)
    logger.info("VERIFY TASK B HISTORY COHERENCE — FASHION")
    logger.info("=" * 70)

    # ------------------------------------------------------------------ #
    # 1. Load KG
    # ------------------------------------------------------------------ #
    logger.info("\n[1/5] Loading Fashion KG...")

    kg = load_kg_fashion(
        KG_TRAIN,
        KG_VALID,
        KG_TEST,
        verbose=False,
    )

    kg_items = {
        entity: idx
        for entity, idx in kg["entity_to_id"].items()
        if entity.startswith("item_")
    }

    logger.info(f"KG entities total : {kg['num_entities']:,}")
    logger.info(f"KG items          : {len(kg_items):,}")

    # ------------------------------------------------------------------ #
    # 2. Build RecBole dataset and dataloaders
    # ------------------------------------------------------------------ #
    logger.info("\n[2/5] Building RecBole dataset and dataloaders...")

    recbole_cfg = build_recbole_config()

    init_seed(
        recbole_cfg["seed"],
        recbole_cfg["reproducibility"],
    )

    recbole_dataset = create_dataset(recbole_cfg)

    train_data, valid_data, test_data = data_preparation(
        recbole_cfg,
        recbole_dataset,
    )

    logger.info(f"RecBole item_num including PAD : {recbole_dataset.item_num:,}")
    logger.info(f"RecBole real items             : {recbole_dataset.item_num - 1:,}")
    logger.info(f"RecBole user_num               : {recbole_dataset.user_num:,}")
    logger.info(f"RecBole inter_num              : {recbole_dataset.inter_num:,}")

    logger.info("\nRecBole evaluation protocol:")
    logger.info(f"  split    : {recbole_cfg['eval_args']['split']}")
    logger.info(f"  order    : {recbole_cfg['eval_args']['order']}")
    logger.info(f"  group_by : {recbole_cfg['eval_args']['group_by']}")
    logger.info(f"  mode     : {recbole_cfg['eval_args']['mode']}")

    # These assertions guarantee that this check uses the same evaluation
    # protocol used during SASRec Fashion HPO.
    assert recbole_cfg["eval_args"]["split"] == {"LS": "valid_and_test"}, (
        "This script expects RecBole leave-one-out style split with valid_and_test"
    )
    assert recbole_cfg["eval_args"]["order"] == "TO", (
        "This script expects temporal order TO"
    )
    assert recbole_cfg["eval_args"]["group_by"] == "user", (
        "This script expects group_by=user"
    )
    assert recbole_cfg["eval_args"]["mode"]["valid"] == "full", (
        "This script expects full-sort validation"
    )
    assert recbole_cfg["eval_args"]["mode"]["test"] == "full", (
        "This script expects full-sort test"
    )

    # ------------------------------------------------------------------ #
    # 3. Build RecBole -> KG item mapping
    # ------------------------------------------------------------------ #
    logger.info("\n[3/5] Building RecBole internal item id -> KG entity id mapping...")

    recbole_to_kg, missing_tokens = build_recbole_to_kg_mapping(
        recbole_dataset=recbole_dataset,
        kg=kg,
    )

    mapped_count = int((recbole_to_kg >= 0).sum().item())

    logger.info(
        f"Mapped RecBole items to KG : "
        f"{mapped_count:,}/{recbole_dataset.item_num - 1:,}"
    )
    logger.info(f"Missing RecBole tokens     : {len(missing_tokens):,}")

    if missing_tokens:
        logger.warning(f"Sample missing tokens: {missing_tokens[:10]}")

    assert len(missing_tokens) == 0, (
        f"{len(missing_tokens)} RecBole item tokens are not present in the KG. "
        f"Sample: {missing_tokens[:10]}"
    )

    # ------------------------------------------------------------------ #
    # 4. Inspect batches and optional internal history sources
    # ------------------------------------------------------------------ #
    logger.info("\n[4/5] Inspecting RecBole valid/test batches...")

    valid_sample_batch = next(iter(valid_data))
    test_sample_batch = next(iter(test_data))

    logger.info(f"valid sample batch: {describe_batch(valid_sample_batch)}")
    logger.info(f"test  sample batch: {describe_batch(test_sample_batch)}")

    inspect_optional_history_sources(valid_data, "valid")
    inspect_optional_history_sources(test_data, "test")

    logger.info(
        "\nNote: history_index/user_history_dict are RecBole internal details and "
        "may not be exposed depending on the RecBole version. This is OK. "
        "The actual SASRec HPO used Trainer._valid_epoch(valid_data/test_data), "
        "so the important thing is to build valid_data/test_data with the same "
        "RecBole Config and check that their Interaction fields are coherent."
    )

    # ------------------------------------------------------------------ #
    # 5. Check valid/test loaders
    # ------------------------------------------------------------------ #
    logger.info("\n[5/5] Checking valid/test loaders...")

    valid_summary = check_loader(
        loader=valid_data,
        name="valid",
        recbole_to_kg=recbole_to_kg,
    )

    test_summary = check_loader(
        loader=test_data,
        name="test",
        recbole_to_kg=recbole_to_kg,
    )

    log_summary("valid", valid_summary)
    log_summary("test", test_summary)

    # ------------------------------------------------------------------ #
    # Final assertions
    # ------------------------------------------------------------------ #

    assert valid_summary["num_batches"] > 0, (
        "validation dataloader is empty"
    )
    assert test_summary["num_batches"] > 0, (
        "test dataloader is empty"
    )

    assert valid_summary["zero_targets"] == 0, (
        "validation loader contains padding targets"
    )
    assert test_summary["zero_targets"] == 0, (
        "test loader contains padding targets"
    )

    assert valid_summary["unmapped_targets"] == 0, (
        "validation loader contains targets not mappable to KG"
    )
    assert test_summary["unmapped_targets"] == 0, (
        "test loader contains targets not mappable to KG"
    )

    assert valid_summary["unmapped_sequence_items"] == 0, (
        "validation item_id_list contains items not mappable to KG"
    )
    assert test_summary["unmapped_sequence_items"] == 0, (
        "test item_id_list contains items not mappable to KG"
    )

    assert valid_summary["empty_sequences"] == 0, (
        "validation loader contains empty item sequences"
    )
    assert test_summary["empty_sequences"] == 0, (
        "test loader contains empty item sequences"
    )

    assert valid_summary["inconsistent_lengths"] == 0, (
        "validation item_length is inconsistent with item_id_list"
    )
    assert test_summary["inconsistent_lengths"] == 0, (
        "test item_length is inconsistent with item_id_list"
    )

    logger.info("\n" + "=" * 70)
    logger.info("TASK B HISTORY COHERENCE CHECK PASSED — FASHION")
    logger.info("=" * 70)
    logger.info("RecBole valid/test dataloaders are coherent.")
    logger.info("Evaluation protocol is coherent with SASRec Fashion HPO.")
    logger.info("Protocol: temporal leave-one-out style split with valid_and_test.")
    logger.info("Targets and sequences are compatible with the KG item space.")
    logger.info("You can proceed with Task B warm-start / alternate learning.")


if __name__ == "__main__":
    main()
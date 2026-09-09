"""
Quick sanity check for Task A type_index shape.

Goal:
- Verify that build_type_index() returns a tensor for type_index['item']
- Verify it is 1D and contains multiple indices
- Keep this check lightweight and independent from training
"""

import os
import sys
import logging
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
ROOT_DIR = os.path.abspath(os.path.join(PARENT_DIR, ".."))

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from load_kg import load_kg

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

KG_TRAIN_PATH = os.path.join(ROOT_DIR, "data", "processed", "kg_train.tsv")
KG_VALID_PATH = os.path.join(ROOT_DIR, "data", "processed", "taskA_valid.tsv")
KG_TEST_PATH = os.path.join(ROOT_DIR, "data", "processed", "taskA_test.tsv")

def build_type_index(kg):
    """Build the type index exactly as in the Task A test utilities."""
    type_to_ids = {'item': [], 'machine': [], 'client': [], 'model': []}
    for label, eid in kg['entity_to_id'].items():
        prefix = label.split('_')[0]
        if prefix in type_to_ids:
            type_to_ids[prefix].append(eid)
    return {k: torch.LongTensor(v) for k, v in type_to_ids.items()}


def main():
    logger.info("Starting Task A type_index sanity check")
    kg = load_kg(
        KG_TRAIN_PATH,
        KG_VALID_PATH,
        KG_TEST_PATH,
    )
    type_index = build_type_index(kg)

    item_ids = type_index['item']
    logger.info(f"type_index['item'] type: {type(item_ids).__name__}")
    logger.info(f"type_index['item'] dtype: {item_ids.dtype}")
    logger.info(f"type_index['item'] ndim: {item_ids.ndim}")
    logger.info(f"type_index['item'] shape: {tuple(item_ids.shape)}")
    logger.info(f"type_index['item'] count: {item_ids.numel()}")
    logger.info(f"First 10 item ids: {item_ids[:10].tolist()}")

    assert isinstance(item_ids, torch.Tensor), "type_index['item'] must be a torch.Tensor"
    assert item_ids.ndim == 1, "type_index['item'] must be a 1D tensor"
    assert item_ids.numel() > 1, "type_index['item'] should contain multiple item indices"
    assert item_ids.dtype == torch.long, "type_index['item'] should be a LongTensor"

    logger.info("Task A type_index sanity check PASSED")


if __name__ == "__main__":
    main()

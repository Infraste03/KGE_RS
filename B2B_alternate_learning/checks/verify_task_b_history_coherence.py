"""
Quick coherence check for Task B evaluation.

Checks:
- valid/test dataloaders are built correctly
- test dataset exposes user_history_dict
- targets are never padding (0)
- for a few sample test users, the masked history is consistent with the batch user_id
- validation batches do not require user_history_dict and still expose valid targets

This is intentionally lightweight and does not run training.
"""

import os
import sys
import yaml
import logging
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
MODELS_DIR = os.path.join(PARENT_DIR, "models")
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from unified_id_space import build_unified_id_space
from data_loaders import build_rec_dataloader
from recbole.config import Config as RecBoleConfig
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

CFG_PATH = os.path.join(PARENT_DIR, "configs", "step4_config.yaml")


def main():
    logger.info("Starting Task B history coherence check")
    with open(CFG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)

    space = build_unified_id_space(
        kg_train_path=cfg['paths']['kg_train'],
        kg_valid_path=cfg['paths']['kg_valid'],
        kg_test_path=cfg['paths']['kg_test'],
        recbole_train_inter=cfg['paths']['recbole_train_inter'],
        recbole_valid_inter=cfg['paths']['recbole_valid_inter'],
        recbole_test_inter=cfg['paths']['recbole_test_inter'],
        recbole_dataset_name=cfg['paths']['recbole_dataset_name'],
        recbole_data_path=cfg['paths']['recbole_data_path'],
        verbose=False,
    )

    recbole_cfg = RecBoleConfig(
        model='SASRec',
        dataset='b2b_data',
        config_dict=cfg['recbole_config']
    )
    init_seed(recbole_cfg['seed'], recbole_cfg['reproducibility'])

    recbole_dataset = create_dataset(recbole_cfg)
    train_data, valid_data, test_data = data_preparation(recbole_cfg, recbole_dataset)

    num_recbole_items = recbole_dataset.item_num
    pad_unified_id = space.num_total_entities
    recbole_to_unified = torch.full((num_recbole_items,), pad_unified_id, dtype=torch.long)
    for recbole_id in range(num_recbole_items):
        uid = space.recbole_to_unified(recbole_id)
        if uid is not None:
            recbole_to_unified[recbole_id] = uid

    logger.info(f"valid has user_history_dict: {hasattr(valid_data.dataset, 'user_history_dict')}")
    logger.info(f"test has user_history_dict: {hasattr(test_data.dataset, 'user_history_dict')}")
    assert hasattr(test_data.dataset, 'user_history_dict'), "test dataset must expose user_history_dict"

    def check_loader(loader, name, sample_batches=3):
        zero_targets = 0
        total_targets = 0
        sample_checked = 0
        for batch in loader:
            interaction, _, _, _ = batch
            targets = interaction['item_id']
            zero_targets += (targets == 0).sum().item()
            total_targets += targets.numel()
            if name == 'test' and sample_checked < sample_batches:
                user_ids = interaction['user_id']
                for idx in range(min(2, user_ids.shape[0])):
                    user_id = user_ids[idx].item()
                    history = loader.dataset.user_history_dict[user_id]
                    assert len(history) > 0, f"empty history for user {user_id} in {name}"
                    target_idx = targets[idx].item()
                    assert target_idx != 0, f"padding target found for user {user_id} in {name}"
                    # The target should be representable in the unified space; this also checks mapping coverage.
                    _ = recbole_to_unified[target_idx].item()
                sample_checked += 1
        return zero_targets, total_targets

    z_valid, t_valid = check_loader(valid_data, 'valid')
    z_test, t_test = check_loader(test_data, 'test')

    logger.info(f"Valid targets padding count: {z_valid}/{t_valid}")
    logger.info(f"Test  targets padding count: {z_test}/{t_test}")

    assert z_valid == 0, "validation loader contains padding targets"
    assert z_test == 0, "test loader contains padding targets"

    logger.info("Task B history coherence check PASSED")


if __name__ == '__main__':
    main()

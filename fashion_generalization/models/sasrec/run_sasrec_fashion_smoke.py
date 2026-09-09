# fashion_generalization/models/sasrec/run_sasrec_fashion_smoke.py
"""
SASRec Fashion — Local smoke test.
1 epoch with lightweight parameters to verify that the full pipeline runs correctly.
embedding_dim fixed at 64 (from the best TransE HPO).
"""

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import SASRec
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
import logging
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(*args, **kwargs)
torch.load = _patched_load

if __name__ == '__main__':

    out_folder = os.path.join("..", "..", "results", "sasrec_smoke")
    os.makedirs(out_folder, exist_ok=True)

    config_dict = {
        # ── Dataset ──────────────────────────────────────────────────────────
        'dataset': 'fashion',
        'data_path': os.path.join("..", "..", "data", "recbole"),
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'TIME_FIELD': 'timestamp',
        'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
        'field_separator': "\t",

        # ── LOO split ────────────────────────────────────────────────────────
        'eval_args': {
            'split': {'LS': 'valid_and_test'},
            'order': 'TO',
            'group_by': 'user',
            'mode': {'valid': 'full', 'test': 'full'},
        },

        # ── Model ────────────────────────────────────────────────────────────
        'hidden_size': 64,          # fixed from TransE HPO
        'MAX_ITEM_LIST_LENGTH': 50,
        'n_layers': 2,
        'n_heads': 2,               # 64 / 2 = 32 — divisible
        'hidden_dropout_prob': 0.3,
        'attn_dropout_prob': 0.1,
        'loss_type': 'CE',
        'train_neg_sample_args': None,  # CE does not use negative sampling

        # ── Training ─────────────────────────────────────────────────────────
        'epochs': 1,                # smoke test
        'train_batch_size': 256,
        'eval_batch_size': 256,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'stopping_step': 5,

        # ── Output ───────────────────────────────────────────────────────────
        'checkpoint_dir': os.path.join(out_folder, 'saved'),
        'metrics': ['Recall', 'NDCG'],
        'topk': [20],
        'valid_metric': 'NDCG@20',

        # ── Reproducibility ──────────────────────────────────────────────────
        'reproducibility': True,
        'seed': 42,
        'use_gpu': False,           # CPU for local testing
    }

    config = Config(model='SASRec', config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = logging.getLogger()

    logger.info("=" * 60)
    logger.info("SASREC FASHION — SMOKE TEST")
    logger.info("=" * 60)
    logger.info(f"  dataset:    {config_dict['data_path']}")
    logger.info(f"  hidden_size: {config_dict['hidden_size']} (fixed from TransE HPO)")
    logger.info(f"  epochs:     {config_dict['epochs']}")

    dataset = create_dataset(config)
    logger.info(f"  Users:       {dataset.user_num}")
    logger.info(f"  Items:       {dataset.item_num}")
    logger.info(f"  Interactions:{dataset.inter_num}")

    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = SASRec(config, train_data.dataset).to(config['device'])
    trainer = Trainer(config, model)

    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=True
    )

    logger.info(f"  Best valid NDCG@20: {best_valid_score:.4f}")

    test_result = trainer.evaluate(test_data)
    logger.info(f"  Test Recall@20: {test_result.get('recall@20', 0):.4f}")
    logger.info(f"  Test NDCG@20:   {test_result.get('ndcg@20', 0):.4f}")
    logger.info("SMOKE TEST COMPLETED")
    
    # ── LOO SPLIT VERIFICATION ───────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("LOO SPLIT VERIFICATION")
    logger.info("=" * 60)

    # Dataset sizes
    logger.info(f"  Total users:        {dataset.user_num - 1:,}")
    logger.info(f"  Total items:        {dataset.item_num - 1:,}")
    logger.info(f"  Total interactions: {dataset.inter_num:,}")

    # Split sizes
    logger.info(f"  Train interactions: {len(train_data.dataset):,}")
    logger.info(f"  Valid interactions: {len(valid_data.dataset):,}")
    logger.info(f"  Test interactions:  {len(test_data.dataset):,}")

    # In LOO, validation and test must contain exactly 1 interaction per user
    n_users_valid = len(set(valid_data.dataset['user_id'].numpy()))
    n_users_test  = len(set(test_data.dataset['user_id'].numpy()))
    logger.info(f"  Users in validation: {n_users_valid:,}")
    logger.info(f"  Users in test:       {n_users_test:,}")

    # Verify LOO: each user must have exactly 1 item in validation and 1 in test
    inter_per_user_valid = valid_data.dataset['user_id'].numpy()
    from collections import Counter
    counts_valid = Counter(inter_per_user_valid)
    max_valid = max(counts_valid.values())
    min_valid = min(counts_valid.values())
    logger.info(f"  Interactions per user in validation: min={min_valid}, max={max_valid}")
    if max_valid == 1 and min_valid == 1:
        logger.info("  OK: correct LOO — 1 item per user in validation")
    else:
        logger.warning("  WARNING: incorrect LOO split in validation")

    inter_per_user_test = test_data.dataset['user_id'].numpy()
    counts_test = Counter(inter_per_user_test)
    max_test = max(counts_test.values())
    min_test = min(counts_test.values())
    logger.info(f"  Interactions per user in test:       min={min_test}, max={max_test}")
    if max_test == 1 and min_test == 1:
        logger.info("  OK: correct LOO — 1 item per user in test")
    else:
        logger.warning("  WARNING: incorrect LOO split in test")

    # Verify consistency with the original .inter file
    import pandas as pd
    # Original file: fashion_generalization/data/recbole/fashion/fashion.inter
    inter_original = pd.read_csv(
        os.path.join("..", "..", "data", "recbole", "fashion", "fashion.inter"),
        sep='\t'
    )
    n_users_original = inter_original['user_id:token'].nunique()
    n_items_original = inter_original['item_id:token'].nunique()
    n_inter_original = len(inter_original)
    logger.info(f"\n  Original .inter file:")
    logger.info(f"    Users:        {n_users_original:,}")
    logger.info(f"    Items:        {n_items_original:,}")
    logger.info(f"    Interactions: {n_inter_original:,}")
    logger.info(f"  RecBole dataset:")
    logger.info(f"    Users:        {dataset.user_num - 1:,}")
    logger.info(f"    Items:        {dataset.item_num - 1:,}")
    logger.info(f"    Interactions: {dataset.inter_num:,}")

    # Excluded users (users with fewer than 3 interactions)
    utenti_esclusi = n_users_original - (dataset.user_num - 1)
    logger.info(f"  Users excluded by RecBole (seq<3): {utenti_esclusi:,}")

    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION COMPLETED")
    logger.info("=" * 60)
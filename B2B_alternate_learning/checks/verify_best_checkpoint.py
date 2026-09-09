"""
verify_best_checkpoint.py

Verify that reloading best_model.pt (the checkpoint saved at the best time
on validation) you get EXACTLY the metrics reported in the HPO run.

Purpose: To confirm, BEFORE launching multi-seed runs, that the evaluation is
deterministic and that the saved checkpoint is valid. If this script returns
R@20 = 0.5000 (or very close value) on the validation, then the results are
reproducible and you can proceed with the 5 seeds.

Usage:
    python B2B_alternate_learning/checks/verify_best_checkpoint.py \
        --config B2B_alternate_learning/configs/step4_seeds_one_to_one.yaml \
        --checkpoint hpc_step4_results/step4_one_to_one_hpo_trial1_seed2020/best_model.pt \
        --seed 2020
"""

import os
import sys
import yaml
import torch
import argparse


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
ROOT_DIR = os.path.dirname(PARENT_DIR)
for p in (PARENT_DIR, ROOT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

MODELS_DIR = os.path.join(PARENT_DIR, 'models')
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from unified_id_space import build_unified_id_space
from joint_model import JointAlternateModel
from data_loaders import build_rec_dataloader
from eval_utils import evaluate_task_b

from recbole.config import Config as RecBoleConfig
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Verify reproducibility of best_model.pt")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path a best_model.pt da verificare")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--expected-valid-r20", type=float, default=None,
                        help="Expected Recall@20 on validation. If provided, the script will check if the actual value is close to this expected value.")
    args = parser.parse_args()

    config = load_config(args.config)
    config['recbole_config']['seed'] = args.seed

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("=" * 70)
    print("VERIFICATION OF BEST CHECKPOINT REPRODUCIBILITY")
    print("=" * 70)
    print(f"Config     : {args.config}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Seed       : {args.seed}")
    print(f"Expected R@20: {args.expected_valid_r20}")

    # --- 1) Unified ID space ---
    print("\n[1] Unified ID Space Construction...")
    space = build_unified_id_space(
        kg_train_path=config['paths']['kg_train'],
        kg_valid_path=config['paths']['kg_valid'],
        kg_test_path=config['paths']['kg_test'],
        recbole_train_inter=config['paths']['recbole_train_inter'],
        recbole_valid_inter=config['paths']['recbole_valid_inter'],
        recbole_test_inter=config['paths']['recbole_test_inter'],
        recbole_dataset_name=config['paths']['recbole_dataset_name'],
        recbole_data_path=config['paths']['recbole_data_path'],
    )
    PAD_UNIFIED_ID = space.num_total_entities

    print(f"PAD_UNIFIED_ID: {PAD_UNIFIED_ID}")

    print("[2] Loading RecBole dataset...")
    recbole_cfg = RecBoleConfig(model='SASRec', dataset='b2b_data',
                                config_dict=config['recbole_config'])
    init_seed(recbole_cfg['seed'], recbole_cfg['reproducibility'])
    recbole_dataset = create_dataset(recbole_cfg)
    train_data, valid_data, test_data = data_preparation(recbole_cfg, recbole_dataset)
    num_recbole_items = recbole_dataset.item_num

    recbole_to_unified_tensor = torch.zeros(num_recbole_items, dtype=torch.long)
    for rid in range(num_recbole_items):
        uid = space.recbole_to_unified(rid)
        recbole_to_unified_tensor[rid] = uid if uid is not None else 0
    recbole_to_unified_tensor = recbole_to_unified_tensor.to(device)


    print("[3] Model instantiation...")
    model = JointAlternateModel(
        num_entities=space.num_total_entities + 1,
        num_relations=5,
        num_items=num_recbole_items,
        kge_dim=config['model']['kge_dim'],
        n_layers=config['model'].get('n_layers', 4),
        n_heads=config['model'].get('n_heads', 2),
        hidden_dropout=config['model'].get('hidden_dropout', 0.4),
        attn_dropout=config['model'].get('attn_dropout', 0.3),
        padding_idx=PAD_UNIFIED_ID,
    ).to(device)


    print("[4] Loading checkpoint...")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if isinstance(state, dict):
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        elif 'model_state' in state:
            state = state['model_state']


    model.load_state_dict(state)
    model.eval()

    print("[5] Evaluation on validation and test sets...\n")
    val_metrics = evaluate_task_b(model, valid_data, recbole_to_unified_tensor)
    test_metrics = evaluate_task_b(model, test_data, recbole_to_unified_tensor)

    v_r20 = val_metrics.get('Recall@20', 0.0)
    v_ndcg = val_metrics.get('NDCG@20', 0.0)
    t_r20 = test_metrics.get('Recall@20', 0.0)
    t_ndcg = test_metrics.get('NDCG@20', 0.0)

    print("=" * 70)
    print("RESULTS OF THE VERIFICATION")
    print("=" * 70)
    print(f"  [Validation] R@20 = {v_r20:.4f}   NDCG@20 = {v_ndcg:.4f}")
    print(f"  [Test]       R@20 = {t_r20:.4f}   NDCG@20 = {t_ndcg:.4f}")

    if args.expected_valid_r20 is not None:
        diff = abs(v_r20 - args.expected_valid_r20)
        print(f"\n  expected (validation R@20): {args.expected_valid_r20:.4f}")
        print(f"  Difference               : {diff:.6f}")
        if diff < 1e-3:
            print("\n  outcome: results are reproducible (diff < 0.001). go on.")
        else:
            print("\n  outcome: attention - difference is significant. Verify seed/masking/config.")
    print("=" * 70)


if __name__ == '__main__':
    main()
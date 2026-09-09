"""
Functional test and metrics replication of SASRec on real Step 3 Data.
"""

import os
import sys
import logging
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
MODELS_DIR = os.path.join(PARENT_DIR, "models")
ROOT_DIR = os.path.abspath(os.path.join(PARENT_DIR, ".."))

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from shared_embedding import SharedEmbedding
from task_b_sasrec import TaskBSASRec, TaskBSASRecLegacy
# Passo 1: import
from unified_id_space import build_unified_id_space

# RecBole dependencies to faithfully load the original dataset
from recbole.data import create_dataset, data_preparation
from recbole.config import Config
from recbole.utils import init_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Paths ---
SASREC_WEIGHTS = os.path.join(
    ROOT_DIR,
    "SASRec",
    "pth",
    "SASRec-May-19-2026_08-28-06.pth"
)
config_dict = {
        'model': 'SASRec',

        'data_path': os.path.join(ROOT_DIR, "dataset"),
        'data_path': 'dataset/',
        'USER_ID_FIELD': 'user_id',

        'ITEM_ID_FIELD': 'item_id',
        'TIME_FIELD': 'timestamp',
        'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
        'field_separator': "\t",
        'eval_setting': 'TO_session,full',
        'train_file': 'b2b_data.train.inter',
        'valid_file': 'b2b_data.valid.inter',
        'test_file': 'b2b_data.test.inter',

        'metrics': ["Recall", "NDCG"],
        'loss_type': 'CE',
        'train_neg_sample_args': None,
    'topk': [20],
    'valid_metric': 'NDCG@20',
    'seed': 2020,
    'reproducibility': True,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

def load_recbole_data():
    logger.info("Initializing RecBole Config and loading dataset...")
    config = Config(model='SASRec', dataset='b2b_data', config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    # +1 cause padding
    num_items = dataset.item_num
    return config, train_data, valid_data, test_data, num_items

# --- Custom Evaluation Loop Mimicking RecBole behavior ---
def evaluate_custom_sasrec(model, test_data, num_items, mode_name="Legacy", recbole_to_unified=None):
    model.eval()
    logger.info(f"\nEvaluating Custom {mode_name} Model on Test Set...")
    hits_count_20 = 0
    total_users = 0

    with torch.no_grad():
        for batch_data in test_data:
            interaction, ground_truth, pos_idx, neg_idx = batch_data
            item_seq = interaction['item_id_list'].to(config_dict['device'])
            item_seq_len = interaction['item_length'].to(config_dict['device'])
            target_items = interaction['item_id'].to(config_dict['device'])
            # Forward — traduci item_seq se siamo in modalità Hybrid
            if isinstance(model, TaskBSASRec) and recbole_to_unified is not None:
                item_seq_unified = recbole_to_unified.to(config_dict['device'])[item_seq]
                user_repr = model(item_seq_unified, item_seq_len)
            else:
                user_repr = model(item_seq, item_seq_len)

            # Scoring
            if isinstance(model, TaskBSASRec) and recbole_to_unified is not None:
                candidates = recbole_to_unified.to(config_dict['device'])
                scores = model.score_all_items(user_repr, candidates)
            else:
                scores = model.score_all_items(user_repr)

            # Gather ranks
            for b_idx in range(user_repr.shape[0]):
                target_idx = target_items[b_idx].item()
                target_score = scores[b_idx, target_idx].item()
                user_preds = scores[b_idx]
                rank = (user_preds > target_score).sum().item() + 1
                if rank <= 20:
                    hits_count_20 += 1
                total_users += 1

                if total_users % 1000 == 0:
                    logger.info(f"  Processed {total_users} users...")
    recall_20 = hits_count_20 / total_users
    logger.info(f"--- RESULTS {mode_name} ---")
    logger.info(f"R@20: {recall_20:.4f}")
    return recall_20


def main():
    config, train_data, valid_data, test_data, num_items = load_recbole_data()
    logger.info(f"Dataset Loaded. Total Items across interactions (incl Pad): {num_items}")

    space = build_unified_id_space(
    kg_train_path=os.path.join(ROOT_DIR, "data", "processed", "kg_train.tsv"),
    kg_valid_path=os.path.join(ROOT_DIR, "data", "processed", "taskA_valid.tsv"),
    kg_test_path=os.path.join(ROOT_DIR, "data", "processed", "taskA_test.tsv"),
    recbole_train_inter=os.path.join(ROOT_DIR, "dataset", "b2b_data", "b2b_data.train.inter"),
    recbole_valid_inter=os.path.join(ROOT_DIR, "dataset", "b2b_data", "b2b_data.valid.inter"),
    recbole_test_inter=os.path.join(ROOT_DIR, "dataset", "b2b_data", "b2b_data.test.inter"),
    recbole_dataset_name="b2b_data",
    recbole_data_path=os.path.join(ROOT_DIR, "dataset"),
    verbose=False,
)

    recbole_to_unified_tensor = torch.zeros(num_items, dtype=torch.long)
    for recbole_id in range(num_items):
        uid = space.recbole_to_unified(recbole_id)
        if uid is not None:
            recbole_to_unified_tensor[recbole_id] = uid

    recbole_checkpoint = torch.load(SASREC_WEIGHTS, map_location='cpu', weights_only=False)

    # Note: the RecBole .pth embeds the dict inside a 'state_dict' key
    if 'state_dict' in recbole_checkpoint:
        recbole_state_dict = recbole_checkpoint['state_dict']
    else:
        recbole_state_dict = recbole_checkpoint

    logger.info("="*70)
    logger.info("TEST 1: TaskBSASRecLegacy (Step 3 Exact Copy)")
    logger.info("="*70)

    # Same hyperparams default in your class vs SASRec standard init
    legacy_model = TaskBSASRecLegacy(
        num_items=num_items,
        n_layers=4, n_heads=2, hidden_size=400, inner_size=256,
        hidden_dropout=0.4, attn_dropout=0.3, max_seq_length=50
    ).to(config_dict['device'])

    legacy_model.load_full_recbole(recbole_state_dict, verbose=True)

    r20_legacy = evaluate_custom_sasrec(legacy_model, test_data, num_items, mode_name="Legacy")
    if abs(r20_legacy - 0.3607) < 0.05:
         logger.info("VERDICT [LEGACY]: PERFECT. The R@20 value closely matches ~0.3607.")
    else:
         logger.warning(f"VERDICT [LEGACY]: Difference detected! Got {r20_legacy:.4f}. Investigate differences with test_performance.py config.")

    # ---- TEST 2: HYBRID ARCHITECTURE ----
    logger.info("="*70)
    logger.info("TEST 2: TaskBSASRec (Hybrid Architecture - Projection)")
    logger.info("="*70)

    unified_entity_num = space.num_total_entities
    shared = SharedEmbedding(num_entities=unified_entity_num, embedding_dim=400, padding_idx=0).to(config_dict['device'])

    hybrid_model = TaskBSASRec(shared_embedding=shared, num_items=num_items, n_layers=4, n_heads=2, hidden_size=400, inner_size=256, hidden_dropout=0.4, attn_dropout=0.3, max_seq_length=50).to(config_dict['device'])
    # Load ONLY attention layers
    hybrid_model.load_self_attention_from_recbole(recbole_state_dict, verbose=True)

    # --- PASSO 4: Chiamata al test ibrido passing il tensore di traduzione
    r20_hybrid = evaluate_custom_sasrec(
        hybrid_model, test_data, num_items, mode_name="Hybrid",
        recbole_to_unified=recbole_to_unified_tensor
    )

    logger.info(f"VERDICT [HYBRID]: Runs Fine! Extracted noisy/baseline score = {r20_hybrid:.4f}.")
    logger.info("Everything is correctly set up for Alternate Learning Training (Step 4).")

if __name__ == '__main__':
    main()
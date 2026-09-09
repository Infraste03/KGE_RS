"""
test_task_b_real_data_fashion.py
==================================

Verify that our Task B infrastructure replicates the metrics of the
SASRec fashion checkpoint (trial 17, best_valid).

Two tests:
  TEST 1 - Legacy: Load ALL weights into TaskBSASRecLegacy (own
           item_embedding) and verify that R@20 ~ 0.0951 (test_from_best_valid_recall).
           This proves that our TransformerEncoder is identical to RecBole.

  TEST 2 - Hybrid: Load only transformer weights into TaskBSASRec
           (SharedEmbedding initialized random). This must NOT be replicated
           the metrics (random embedding), but it must run without errors and
           produce sensible scores. It is the starting point of alternate learning.

"""

import os
import sys
import logging
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import math

THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
MODELS_DIR = os.path.join(PARENT_DIR, "models")
BASE_DIR   = os.path.abspath(os.path.join(PARENT_DIR, ".."))


if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from shared_embedding_fashion import SharedEmbedding
from task_b_sasrec_fashion import TaskBSASRec, TaskBSASRecLegacy
from unified_id_space_fashion import build_unified_id_space_fashion

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Paths e config
# --------------------------------------------------------------------------- #
SASREC_CKPT  = os.path.join(BASE_DIR, "results", "sasrec_hpo", "trial_017", "best_valid.pth")
RECBOLE_DATA = os.path.join(BASE_DIR, "data", "recbole")

EXPECTED_RECALL_20 = 0.0951
EXPECTED_NDCG_20   = 0.0889

RECBOLE_CONFIG = {
    'data_path'      : RECBOLE_DATA,
    'USER_ID_FIELD'  : 'user_id',
    'ITEM_ID_FIELD'  : 'item_id',
    'TIME_FIELD'     : 'timestamp',
    'load_col'       : {'inter': ['user_id', 'item_id', 'timestamp']},
    'field_separator': "\t",
    'eval_args': {
        'split'   : {'LS': 'valid_and_test'},
        'order'   : 'TO',
        'group_by': 'user',
        'mode'    : {'valid': 'full', 'test': 'full'},
    },
    'MAX_ITEM_LIST_LENGTH': 50,
    'hidden_size'   : 64,
    'n_layers'      : 4,
    'n_heads'       : 4,
    'inner_size'    : 256,
    'hidden_dropout_prob': 0.5,
    'attn_dropout_prob'  : 0.3,
    'loss_type'     : 'CE',
    'train_neg_sample_args': None,
    'metrics'       : ['Recall', 'NDCG'],
    'topk'          : [20],
    'valid_metric'  : 'NDCG@20',
    'seed'          : 2020,
    'reproducibility': True,
    'show_progress' : False,
    'use_gpu'       : True,
}

N_LAYERS       = 4
N_HEADS        = 4
HIDDEN_SIZE    = 64
INNER_SIZE     = 256
HIDDEN_DROPOUT = 0.5
ATTN_DROPOUT   = 0.3
MAX_SEQ_LEN    = 50

NUM_KG_ENTITIES = 182_983
PADDING_IDX     = 182_983
NUM_ROWS        = 182_984
ITEM_ID_START   = 17_514



def load_recbole_data():
    logger.info("Loading RecBole fashion dataset...")
    config = Config(model='SASRec', dataset='fashion', config_dict=RECBOLE_CONFIG)
    init_seed(config['seed'], config['reproducibility'])
    dataset = create_dataset(config)
    _, valid_data, test_data = data_preparation(config, dataset)
    num_items = dataset.item_num  # 165,470
    logger.info(f"  item_num (con PAD): {num_items:,}")
    return config, test_data, num_items


def load_checkpoint():
    ckpt = torch.load(SASREC_CKPT, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        return ckpt['state_dict']
    return ckpt


def build_recbole_to_unified_tensor(space, num_items):
    """
    Constructs the translation tensor RecBole internal id -> unified id.
    Unmapped locations (padding RecBole id=0) receive PADDING_IDX.
    """
    tensor = torch.full((num_items,), PADDING_IDX, dtype=torch.long)
    for recbole_id in range(num_items):
        uid = space.recbole_to_unified(recbole_id)
        if uid is not None:
            tensor[recbole_id] = uid
    return tensor


def evaluate_with_recbole_trainer(state_dict, config, test_data):
    """
    Evaluate the checkpoint using the official RecBole Trainer.
    This is the ground truth: whatever metric comes out here is that
    that you need to replicate in your custom loop.
    """
    from recbole.model.sequential_recommender import SASRec
    from recbole.trainer import Trainer

    logger.info("\n--- Ground truth RecBole Trainer ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SASRec(config, test_data.dataset).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    trainer = Trainer(config, model)
    _, test_result = trainer._valid_epoch(test_data, show_progress=False)

    logger.info(f"  RecBole Trainer Recall@20 : {test_result.get('recall@20', 'N/A'):.4f}")
    logger.info(f"  RecBole Trainer NDCG@20   : {test_result.get('ndcg@20',   'N/A'):.4f}")
    return test_result



def evaluate(model, test_data, mode, recbole_to_unified=None):
    """
    Evaluation loop that replicates the behavior of RecBole Trainer.


    Parameters
    ----------
    model : TaskBSASRecLegacy o TaskBSASRec
    test_data : RecBole dataloader
    mode : 'legacy' o 'hybrid'
    recbole_to_unified : LongTensor (num_items,), solo per mode='hybrid'

    Returns
    -------
    recall_20, ndcg_20 : float
    """
    model.eval()
    device = next(model.parameters()).device

    hits_20   = 0
    ndcg_20   = 0.0
    total     = 0

    with torch.no_grad():
        for batch in test_data:
            interaction, _, _, _ = batch

            item_seq     = interaction['item_id_list'].to(device)
            item_seq_len = interaction['item_length'].to(device)
            targets      = interaction['item_id'].to(device)   # RecBole raw IDs

            if mode == 'legacy':
                # item_seq contiene RecBole raw IDs, padding=0
                user_repr = model(item_seq, item_seq_len)
                scores    = model.score_all_items(user_repr)
                for b in range(user_repr.shape[0]):
                    target_id = targets[b].item()
                    seq_len   = item_seq_len[b].item()

                    scores_b = scores[b].clone()
                    for pos in range(seq_len - 1):   #  seq_len-1, non seq_len
                        sid = item_seq[b, pos].item()
                        if sid != 0:
                            scores_b[sid] = float('-inf')

                    target_score = scores_b[target_id].item()
                    rank = (scores_b > target_score).sum().item() + 1
                    if rank <= 20:
                        hits_20 += 1
                        ndcg_20 += 1.0 / math.log2(rank + 1)
                    total += 1


            else:  # hybrid
                r2u = recbole_to_unified.to(device)
                item_seq_unified = r2u[item_seq]
                user_repr = model(item_seq_unified, item_seq_len)

                candidate_ids = torch.arange(
                    ITEM_ID_START, NUM_KG_ENTITIES, dtype=torch.long, device=device
                )
                scores = model.score_all_items(user_repr, candidate_ids)

                for b in range(user_repr.shape[0]):
                    target_recbole_id = targets[b].item()
                    target_unified_id = r2u[target_recbole_id].item()
                    target_pos = target_unified_id - ITEM_ID_START
                    if target_pos < 0 or target_pos >= scores.shape[1]:
                        total += 1
                        continue
                    seq_len = item_seq_len[b].item()

                    scores_b = scores[b].clone()
                    for pos in range(seq_len - 1):
                        sid = item_seq[b, pos].item()
                        if sid == 0:
                            continue
                        seen_unified = r2u[sid].item()
                        seen_pos = seen_unified - ITEM_ID_START
                        if 0 <= seen_pos < scores_b.shape[0]:
                            scores_b[seen_pos] = float('-inf')

                    target_score = scores_b[target_pos].item()
                    rank = (scores_b > target_score).sum().item() + 1
                    if rank <= 20:
                        hits_20 += 1
                        ndcg_20 += 1.0 / math.log2(rank + 1)
                    total += 1
                    if total % 5000 == 0:
                        logger.info(f"  [{mode}] {total:,} processed users...")


    recall = hits_20 / total
    ndcg   = ndcg_20 / total
    return recall, ndcg



def main():
    logger.info("=" * 70)
    logger.info("TEST TASK B REAL DATA — FASHION")
    logger.info("=" * 70)


    config, test_data, num_items = load_recbole_data()
    state_dict = load_checkpoint()
    logger.info(f"  Checkpoint loaded: {len(state_dict)} keys")

    ground_truth = evaluate_with_recbole_trainer(state_dict, config, test_data)
    EXPECTED_RECALL_20 = ground_truth.get('recall@20')
    EXPECTED_NDCG_20   = ground_truth.get('ndcg@20')
    logger.info(f"   Updated expected values: R@20={EXPECTED_RECALL_20:.4f}, NDCG@20={EXPECTED_NDCG_20:.4f}")

    logger.info("\nCostruzione spazio ID unificato...")
    space = build_unified_id_space_fashion(
        kg_train_path=os.path.join(BASE_DIR, "data", "processed", "kg_train.tsv"),
        kg_valid_path=os.path.join(BASE_DIR, "data", "processed", "taskA_valid.tsv"),
        kg_test_path=os.path.join(BASE_DIR, "data", "processed", "taskA_test.tsv"),
        recbole_inter_path=os.path.join(BASE_DIR, "data", "recbole", "fashion", "fashion.inter"),
        recbole_data_path=RECBOLE_DATA,
        verbose=False,
    )

    r2u = build_recbole_to_unified_tensor(space, num_items)
    logger.info(f"  r2u tensor: shape={r2u.shape}, "
                f"padding entries (={PADDING_IDX}): {(r2u == PADDING_IDX).sum().item()}")


    logger.info("\n" + "=" * 70)
    logger.info("TEST 1: TaskBSASRecLegacy - exact replica RecBole")
    logger.info("=" * 70)

    legacy = TaskBSASRecLegacy(
        num_items=num_items,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        hidden_size=HIDDEN_SIZE,
        inner_size=INNER_SIZE,
        hidden_dropout=HIDDEN_DROPOUT,
        attn_dropout=ATTN_DROPOUT,
        max_seq_length=MAX_SEQ_LEN,
    )

    legacy.load_full_recbole(state_dict, verbose=True)

    logger.info("\n  Check for weights loaded in Legacy...")
    ckpt_pos = state_dict['position_embedding.weight']
    our_pos  = legacy.position_embedding.weight.data
    diff_pos = (ckpt_pos - our_pos).abs().max().item()
    assert diff_pos < 1e-6, f"position_embedding does not match! max_diff={diff_pos:.2e}"
    logger.info(f"  position_embedding: max_diff={diff_pos:.2e}")

    ckpt_q = state_dict['trm_encoder.layer.0.multi_head_attention.query.weight']
    our_q  = legacy.trm_encoder.layer[0].multi_head_attention.query.weight.data
    diff_q = (ckpt_q - our_q).abs().max().item()
    assert diff_q < 1e-6, f"query weight layer 0 does not match! max_diff={diff_q:.2e}"
    logger.info(f"  trm_encoder layer 0 query: max_diff={diff_q:.2e}")

    ckpt_ie = state_dict['item_embedding.weight']
    our_ie  = legacy.item_embedding.weight.data
    diff_ie = (ckpt_ie - our_ie).abs().max().item()
    assert diff_ie < 1e-6, f"item_embedding does not match! max_diff={diff_ie:.2e}"
    logger.info(f"  item_embedding: max_diff={diff_ie:.2e}")
    logger.info("  PASS: All weights loaded correctly in Legacy")

    recall_legacy, ndcg_legacy = evaluate(legacy, test_data, mode='legacy')
    logger.info(f"\n  Recall@20 got : {recall_legacy:.4f}")
    logger.info(f"  NDCG@20   got : {ndcg_legacy:.4f}")
    logger.info(f"  Recall@20 expected   : {EXPECTED_RECALL_20:.4f}")
    logger.info(f"  NDCG@20   expected   : {EXPECTED_NDCG_20:.4f}")

    diff_r = abs(recall_legacy - EXPECTED_RECALL_20)
    diff_n = abs(ndcg_legacy   - EXPECTED_NDCG_20)

    # DOPO (più stringente):
    if diff_r < 0.001 and diff_n < 0.001:
        logger.info("  VERDICT [LEGACY]: PERFECT PASS - exact replica RecBole (diff < 0.001).")
    elif diff_r < 0.003 and diff_n < 0.003:
        logger.warning(f"  VERDICT [LEGACY]: PASS with small variation (diff R={diff_r:.4f}, N={diff_n:.4f}).")
        logger.warning("  Causa probabile: virgola mobile / seme. OK if diff < 0.003.")
    else:
        logger.error(f"  VERDICT [LEGACY]: FAIL — diff R@20={diff_r:.4f}, NDCG={diff_n:.4f}")
        logger.error("  STOP: Do not proceed with joint_model_fashion.py until resolved.")


    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: TaskBSASRec Hybrid — SharedEmbedding random + Transformer warm-started")
    logger.info("=" * 70)
    logger.info("NOTE: With random SharedEmbedding the metrics will be low.")
    logger.info("This test only verifies that the infrastructure is running correctly.")
    logger.info("Real metrics will emerge after alternate learning training.")

    shared = SharedEmbedding(
        num_entities=NUM_ROWS,
        embedding_dim=HIDDEN_SIZE,
        padding_idx=PADDING_IDX,
    )

    hybrid = TaskBSASRec(
        shared_embedding=shared,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        hidden_size=HIDDEN_SIZE,
        inner_size=INNER_SIZE,
        hidden_dropout=HIDDEN_DROPOUT,
        attn_dropout=ATTN_DROPOUT,
        max_seq_length=MAX_SEQ_LEN,
        unified_padding_idx=PADDING_IDX,
    )
    hybrid.load_self_attention_from_recbole(state_dict, verbose=True)

    logger.info("\n Check weights loaded into the Hybrid...")
    our_pos_h = hybrid.position_embedding.weight.data
    diff_pos_h = (ckpt_pos - our_pos_h).abs().max().item()
    assert diff_pos_h < 1e-6, f"Hybrid: position_embedding does not match! max_diff={diff_pos_h:.2e}"
    logger.info(f"  Hybrid position_embedding: max_diff={diff_pos_h:.2e} ")


    shared_item_0 = hybrid.shared_embedding.embedding.weight.data[17514]
    ckpt_item_0   = state_dict['item_embedding.weight'][1]
    diff_item_h   = (shared_item_0 - ckpt_item_0).abs().max().item()
    logger.info(f"  Hybrid item_embedding[0] vs ckpt: diff={diff_item_h:.4f} (expected > 0, it's random)")
    assert diff_item_h > 0.01, "SharedEmbedding seems to have loaded item_embedding from RecBole! It should be random for warm-start."
    logger.info("  PASS: SharedEmbedding item embedding is random (ok for warm-start TransE)")

    recall_hybrid, ndcg_hybrid = evaluate(hybrid, test_data, mode='hybrid',
                                           recbole_to_unified=r2u)
    logger.info(f"\n  Recall@20 hybrid (embedding random): {recall_hybrid:.4f}")
    logger.info(f"  NDCG@20   hybrid (embedding random): {ndcg_hybrid:.4f}")
    logger.info(f"  (Baseline random expected: ~1/165469 = {1/165469:.6f})")

    if recall_hybrid > 1.0 / 165_469:
        logger.info("  VERDICT [HYBRID]: PASS — recall > random baseline, infrastructure OK.")
    else:
        logger.warning("  VERDICT [HYBRID]: recall <= random baseline, check the infrastructure.")

    logger.info("\n" + "=" * 70)
    logger.info("DONE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
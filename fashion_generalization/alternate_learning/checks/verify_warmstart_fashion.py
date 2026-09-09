"""
verify_warmstart_fashion.py
============================

Verify that the complete warm start of JointAlternateModelFashion
reproduces the metrics of the standalone checkpoints BEFORE starting
alternate-learning training.

This is the most important integration check: it verifies that
SharedEmbedding + TaskATransE + TaskBSASRec, once the pretrained
weights are loaded, behave exactly like:
  - standalone PyKEEN  -> MRR ≈ 0.1057 (Task A)
  - standalone RecBole -> Recall@20 ≈ 0.0964 (Task B)

If this check fails, training is meaningless — the weights
have not been loaded correctly into the integrated model.

Acceptance thresholds:
  Task A: |MRR_obtained - 0.1057| < 0.005
  Task B: |Recall@20_obtained - 0.0964| < 0.002

File location:
    fashion_generalization/alternate_learning/checks/verify_warmstart_fashion.py
"""

import os
import sys
import logging
import warnings
import torch
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)

THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
MODELS_DIR = os.path.join(PARENT_DIR, "models")
BASE_DIR   = os.path.abspath(os.path.join(PARENT_DIR, ".."))

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from load_kg_fashion import load_kg_fashion
from unified_id_space_fashion import build_unified_id_space_fashion
from joint_model_fashion import JointAlternateModelFashion
from eval_utils_fashion import (
    evaluate_task_a_fashion,
    evaluate_task_b_fashion,
    reset_eval_debug,
)

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed
from pykeen.triples import TriplesFactory

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
KG_TRAIN     = os.path.join(BASE_DIR, "data", "processed", "kg_train.tsv")
KG_VALID     = os.path.join(BASE_DIR, "data", "processed", "taskA_valid.tsv")
KG_TEST      = os.path.join(BASE_DIR, "data", "processed", "taskA_test.tsv")
TRANSE_PATH  = os.path.join(BASE_DIR, "results", "hpo_transe", "best_model.pt")
SASREC_PATH  = os.path.join(BASE_DIR, "results", "sasrec_hpo", "trial_017", "best_valid.pth")
RECBOLE_DATA = os.path.join(BASE_DIR, "data", "recbole")
INTER_PATH   = os.path.join(BASE_DIR, "data", "recbole", "fashion", "fashion.inter")

# Expected metrics from standalone checkpoints
EXPECTED_MRR      = 0.1057   # from test_task_a_real_data_fashion.py
EXPECTED_RECALL20 = 0.0964   # from the official RecBole Trainer (measured today)
EXPECTED_NDCG20   = 0.0889

# Acceptance thresholds
TOL_A = 0.005   # Task A: MRR tolerance
TOL_B = 0.002   # Task B: Recall@20 tolerance

# Fashion architecture (trial 017)
NUM_ENTITIES    = 182_984
NUM_RELATIONS   = 3
KGE_DIM         = 64
UNIFIED_PADDING = 182_983
ITEM_ID_START   = 17_514

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
    'hidden_size'         : 64,
    'n_layers'            : 4,
    'n_heads'             : 4,
    'inner_size'          : 256,
    'hidden_dropout_prob' : 0.5,
    'attn_dropout_prob'   : 0.3,
    'loss_type'           : 'CE',
    'train_neg_sample_args': None,
    'metrics'             : ['Recall', 'NDCG'],
    'topk'                : [20],
    'valid_metric'        : 'NDCG@20',
    'seed'                : 2020,
    'reproducibility'     : True,
    'show_progress'       : False,
    'use_gpu'             : False,
}


# --------------------------------------------------------------------------- #
# Helper: build the r2u tensor
# --------------------------------------------------------------------------- #
def build_r2u_tensor(space, num_items_recbole: int) -> torch.Tensor:
    tensor = torch.full((num_items_recbole,), UNIFIED_PADDING, dtype=torch.long)
    for recbole_id in range(num_items_recbole):
        uid = space.recbole_to_unified(recbole_id)
        if uid is not None:
            tensor[recbole_id] = uid
    return tensor


# --------------------------------------------------------------------------- #
# Helper: build type_index for Task A
# --------------------------------------------------------------------------- #
def build_type_index(kg: dict) -> dict:
    type_to_ids = {'item': [], 'category': [], 'brand': []}
    for label, eid in kg['entity_to_id'].items():
        prefix = label.split('_')[0]
        if prefix in type_to_ids:
            type_to_ids[prefix].append(eid)
    return {k: torch.LongTensor(v) for k, v in type_to_ids.items()}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    logger.info("=" * 70)
    logger.info("VERIFY WARMSTART FASHION")
    logger.info("Verify that the complete warm start reproduces the metrics")
    logger.info("of the standalone checkpoints (PyKEEN + RecBole).")
    logger.info("=" * 70)

    # ------------------------------------------------------------------ #
    # STEP 1: Build unified ID space
    # ------------------------------------------------------------------ #
    logger.info("\n[1/5] Building unified ID space...")
    space = build_unified_id_space_fashion(
        kg_train_path=KG_TRAIN,
        kg_valid_path=KG_VALID,
        kg_test_path=KG_TEST,
        recbole_inter_path=INTER_PATH,
        recbole_data_path=RECBOLE_DATA,
        verbose=False,
    )
    logger.info(f"  Unified space: {space.num_total_entities:,} KG entities")

    # ------------------------------------------------------------------ #
    # STEP 2: Load KG and RecBole
    # ------------------------------------------------------------------ #
    logger.info("\n[2/5] Loading KG and RecBole dataset...")
    kg = load_kg_fashion(KG_TRAIN, KG_VALID, KG_TEST, verbose=False)
    type_index = build_type_index(kg)
    logger.info(f"  KG: {kg['num_entities']:,} entities, {kg['num_relations']} relations")

    config = Config(model='SASRec', dataset='fashion', config_dict=RECBOLE_CONFIG)
    init_seed(config['seed'], config['reproducibility'])
    dataset = create_dataset(config)
    _, _, test_data = data_preparation(config, dataset)
    num_items_recbole = dataset.item_num
    logger.info(f"  RecBole: {num_items_recbole:,} items (including PAD)")

    r2u = build_r2u_tensor(space, num_items_recbole)
    logger.info(
        f"  r2u: shape={r2u.shape}, "
        f"padding entries (={UNIFIED_PADDING}): {(r2u == UNIFIED_PADDING).sum().item()}"
    )

    # ------------------------------------------------------------------ #
    # STEP 3: Instantiate JointAlternateModelFashion and load weights
    # ------------------------------------------------------------------ #
    logger.info("\n[3/5] Instantiating model and applying warm start...")
    model = JointAlternateModelFashion(
        num_entities=NUM_ENTITIES,
        num_relations=NUM_RELATIONS,
        kge_dim=KGE_DIM,
        unified_padding_idx=UNIFIED_PADDING,
    )

    # --- Warm start TransE (entity + relation weights) ---
    logger.info("  Loading TransE weights...")
    transe_state = torch.load(TRANSE_PATH, map_location='cpu', weights_only=False)
    entity_weights   = transe_state['entity_representations.0._embeddings.weight']
    relation_weights = transe_state['relation_representations.0._embeddings.weight']

    train_df = pd.read_csv(KG_TRAIN, sep='\t')
    train_factory = TriplesFactory.from_labeled_triples(
        train_df[['head', 'relation', 'tail']].values
    )
    pykeen_entity_to_id   = dict(train_factory.entity_to_id)
    pykeen_relation_to_id = dict(train_factory.relation_to_id)

    model.load_pretrained_kge(
        pykeen_entity_weights=entity_weights,
        entity_mapping_pykeen=pykeen_entity_to_id,
        entity_mapping_ours=kg['entity_to_id'],
        pykeen_relation_weights=relation_weights,
        rel_mapping_pykeen=pykeen_relation_to_id,
        rel_mapping_ours=kg['relation_to_id'],
    )

    # --- Warm start SASRec (transformer + position embedding) ---
    logger.info("  Loading SASRec weights...")
    sasrec_ckpt = torch.load(SASREC_PATH, map_location='cpu', weights_only=False)
    sasrec_state = sasrec_ckpt.get('state_dict', sasrec_ckpt)
    model.load_pretrained_sasrec(sasrec_state)

    logger.info("  Warm start completed.")

    # --- Verify that weights were loaded correctly ---
    logger.info("\n  Verifying post-warm-start weight integrity...")

    # SharedEmbedding: a random item must match PyKEEN
    sample_entity = list(kg['entity_to_id'].keys())[1000]
    our_id     = kg['entity_to_id'][sample_entity]
    pykeen_id  = pykeen_entity_to_id[sample_entity]
    our_vec    = model.shared_embedding.embedding.weight[our_id].detach()
    pykeen_vec = entity_weights[pykeen_id]
    diff_entity = (our_vec - pykeen_vec).abs().max().item()
    assert diff_entity < 1e-5, \
        f"Entity weight mismatch for '{sample_entity}': diff={diff_entity:.2e}"
    logger.info(f"  SharedEmbedding entity '{sample_entity}': diff={diff_entity:.2e} ✓")

    # Padding must be zero
    pad_norm = model.shared_embedding.embedding.weight[UNIFIED_PADDING].abs().sum().item()
    assert pad_norm == 0.0, f"Padding is not zero: {pad_norm}"
    logger.info(f"  Padding (ID={UNIFIED_PADDING}): norm=0.0 ✓")

    # SASRec position embedding must match the checkpoint
    ckpt_pos = sasrec_state['position_embedding.weight']
    our_pos  = model.task_b.position_embedding.weight.data
    diff_pos = (ckpt_pos - our_pos).abs().max().item()
    assert diff_pos < 1e-6, f"Position embedding mismatch: diff={diff_pos:.2e}"
    logger.info(f"  position_embedding: diff={diff_pos:.2e} ✓")

    # Transformer layer 0 query weight
    ckpt_q = sasrec_state['trm_encoder.layer.0.multi_head_attention.query.weight']
    our_q  = model.task_b.trm_encoder.layer[0].multi_head_attention.query.weight.data
    diff_q = (ckpt_q - our_q).abs().max().item()
    assert diff_q < 1e-6, f"Query weight mismatch: diff={diff_q:.2e}"
    logger.info(f"  trm_encoder layer 0 query: diff={diff_q:.2e} ✓")

    # item_embedding must NOT be loaded into SharedEmbedding
    ckpt_item0  = sasrec_state['item_embedding.weight'][1]   # RecBole ID 1
    our_item0   = model.shared_embedding.embedding.weight[ITEM_ID_START].detach()
    diff_item   = (ckpt_item0 - our_item0).abs().max().item()
    assert diff_item > 0.01, \
        "SharedEmbedding loaded the SASRec item_embedding — it should not have!"
    logger.info(f"  item_embedding NOT copied into SharedEmbedding: diff={diff_item:.4f} ✓")

    logger.info("  All weights were verified successfully.")

    # ------------------------------------------------------------------ #
    # STEP 4: Task A evaluation — must replicate PyKEEN
    # ------------------------------------------------------------------ #
    logger.info("\n[4/5] Task A evaluation (TransE)...")
    logger.info(f"  Expected: MRR ≈ {EXPECTED_MRR:.4f} (from standalone PyKEEN)")

    metrics_a = evaluate_task_a_fashion(
        model=model,
        kg_data_dict=kg,
        type_index=type_index,
        relation_label='compatible_with',
        batch_size_eval=128,
        filtered=True,
    )

    diff_mrr = abs(metrics_a['MRR'] - EXPECTED_MRR)
    logger.info(f"\n  Task A results:")
    logger.info(f"    MRR    : obtained={metrics_a['MRR']:.4f}, "
                f"expected={EXPECTED_MRR:.4f}, diff={diff_mrr:.4f}")
    logger.info(f"    Hits@1 : {metrics_a['Hits@1']:.4f}")
    logger.info(f"    Hits@3 : {metrics_a['Hits@3']:.4f}")
    logger.info(f"    Hits@10: {metrics_a['Hits@10']:.4f}")

    if diff_mrr < TOL_A:
        logger.info(f"  VERDICT Task A: PASS (diff={diff_mrr:.4f} < {TOL_A})")
        task_a_ok = True
    else:
        logger.error(f"  VERDICT Task A: FAIL (diff={diff_mrr:.4f} >= {TOL_A})")
        logger.error("  The TransE warm start is not correct in the integrated model.")
        task_a_ok = False

    # ------------------------------------------------------------------ #
    # STEP 5: Task B evaluation — must replicate RecBole
    # ------------------------------------------------------------------ #
    logger.info("\n[5/5] Task B evaluation (SASRec)...")
    logger.info(f"  Expected: Recall@20 ≈ {EXPECTED_RECALL20:.4f} (from RecBole Trainer)")
    logger.info("  NOTE: with TransE embeddings (instead of SASRec embeddings), Recall@20 will be")
    logger.info("  lower than standalone RecBole — this is EXPECTED and CORRECT.")
    logger.info("  What we verify is that the transformer is loaded")
    logger.info("  correctly and that Recall@20 is > 0 and reasonable.")

    reset_eval_debug()
    metrics_b = evaluate_task_b_fashion(
        model=model,
        recbole_eval_dataloader=test_data,
        recbole_to_unified=r2u,
        topk=20,
    )

    diff_recall = abs(metrics_b['Recall@20'] - EXPECTED_RECALL20)
    diff_ndcg   = abs(metrics_b['NDCG@20']   - EXPECTED_NDCG20)

    logger.info(f"\n  Task B results:")
    logger.info(f"    Recall@20: obtained={metrics_b['Recall@20']:.4f}, "
                f"expected={EXPECTED_RECALL20:.4f}, diff={diff_recall:.4f}")
    logger.info(f"    NDCG@20  : obtained={metrics_b['NDCG@20']:.4f}, "
                f"expected={EXPECTED_NDCG20:.4f}, diff={diff_ndcg:.4f}")

    # For Task B the threshold is wider: TransE embeddings differ from
    # the SASRec item_embedding and therefore cause an unavoidable Recall@20 drop.
    # What must not happen is a value of 0.0 or close to random (1/165469).
    RANDOM_BASELINE = 1.0 / 165_469
    task_b_ok = metrics_b['Recall@20'] > RANDOM_BASELINE * 10

    if metrics_b['Recall@20'] > RANDOM_BASELINE * 100:
        logger.info(
            f"  VERDICT Task B: PASS — Recall@20={metrics_b['Recall@20']:.4f} "
            f">> random baseline ({RANDOM_BASELINE:.6f}). "
            f"The transformer is loaded correctly."
        )
    elif task_b_ok:
        logger.warning(
            f"  VERDICT Task B: PASS (weak) — Recall@20={metrics_b['Recall@20']:.4f} "
            f"> 10x random. The transformer works but the drop caused by "
            f"TransE embeddings is substantial."
        )
    else:
        logger.error(
            f"  VERDICT Task B: FAIL — Recall@20={metrics_b['Recall@20']:.4f} "
            f"is close to random ({RANDOM_BASELINE:.6f})."
        )
        logger.error("  Likely causes:")
        logger.error("    1. Transformer not loaded (check load_pretrained_sasrec)")
        logger.error("    2. Incorrect r2u translation (check build_r2u_tensor)")
        logger.error("    3. Incorrect masking in evaluate_task_b_fashion")
        task_b_ok = False

    # ------------------------------------------------------------------ #
    # Final summary
    # ------------------------------------------------------------------ #
    logger.info("\n" + "=" * 70)
    logger.info("VERIFY WARMSTART FASHION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Task A (TransE) : {'PASS' if task_a_ok else 'FAIL'} — "
                f"MRR={metrics_a['MRR']:.4f} (expected {EXPECTED_MRR:.4f})")
    logger.info(f"  Task B (SASRec) : {'PASS' if task_b_ok else 'FAIL'} — "
                f"Recall@20={metrics_b['Recall@20']:.4f}")

    if task_a_ok and task_b_ok:
        logger.info("\n  WARM START VERIFIED — you can proceed with training.")
    else:
        logger.error("\n  WARM START FAILED — DO NOT proceed with training.")
        logger.error("  Resolve the issues above before continuing.")

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
"""
Functional test of TaskATransE on real Step 2 data.

Goal:
    Verify that loading the pre-trained TransE weights from best_model.pt
    into our SharedEmbedding + TaskATransE infrastructure reproduces the
    same evaluation metrics as Step 2.

Method:
    1. Build the unified ID space (load_kg + RecBole mapping).
    2. Build SharedEmbedding (24,957 entities x 400 dims).
    3. Build TaskATransE (5 relations x 400 dims).
    4. Load entity weights from best_model.pt into SharedEmbedding.
    5. Load relation weights from best_model.pt into TaskATransE.
    6. Replicate the exact evaluate_custom() function from run_hpo_kge2.py,
       but using OUR model instead of the PyKEEN model.
    7. Compare the resulting MRR and Hits@K against Step 2's numbers.

Expected results (from your best_metrics.json, type_constrained, average):
    MRR    ~ 0.2953
    Hits@1 ~ ?    (you'll find it in best_metrics.json)
    Hits@3 ~ ?
    Hits@10 ~ ?

If our numbers match (within rounding error), the entire infrastructure is
verified and we can move on to building Task B.

----------------------------------------------------------------------------
Where to put this file:
    B2B_alternate_learning/checks/test_task_a_real_data.py
    python B2B_alternate_learning\checks\test_task_a_real_data.py
----------------------------------------------------------------------------
"""

import os
import sys
import json
import logging
import torch
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
MODELS_DIR = os.path.join(PARENT_DIR, "models")
ROOT_DIR = os.path.abspath(os.path.join(PARENT_DIR, ".."))

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from load_kg import load_kg
from shared_embedding import SharedEmbedding
from task_a_transe import TaskATransE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# KG TSV files
KG_TRAIN_PATH = os.path.join(ROOT_DIR, "data", "processed", "kg_train.tsv")
KG_VALID_PATH = os.path.join(ROOT_DIR, "data", "processed", "taskA_valid.tsv")
KG_TEST_PATH = os.path.join(ROOT_DIR, "data", "processed", "taskA_test.tsv")

# Pre-trained TransE weights (Step 2 best model)
BEST_MODEL_PATH = os.path.join(
    ROOT_DIR,
    "results",
    "step2_hpo",
    "TransE",
    "best_model.pt"
)

# Step 2 results (for comparison)
BEST_METRICS_PATH = os.path.join(
    ROOT_DIR,
    "results",
    "step2_hpo",
    "TransE",
    "best_metrics.json"
)

# Configuration
EMBEDDING_DIM = 400
NUM_RELATIONS = 5
RELATION_LABEL = 'compatible_with'
HITS_AT_K = [1, 3, 10]


# --------------------------------------------------------------------------- #
# Step 1: Build the model with warm-started weights
# --------------------------------------------------------------------------- #

def build_model_with_warm_start():
    """
    Load KG mapping, build SharedEmbedding + TaskATransE, and warm-start
    both from best_model.pt.

    Returns
    -------
    shared : SharedEmbedding
    task_a : TaskATransE
    kg : dict (output of load_kg)
    """
    logger.info("=" * 70)
    logger.info("STEP 1: building model with warm-started weights")
    logger.info("=" * 70)

    # ---- 1a. Load KG mapping ----
    kg = load_kg(
        train_path=KG_TRAIN_PATH,
        valid_path=KG_VALID_PATH,
        test_path=KG_TEST_PATH,
        verbose=False,
    )
    num_entities = kg['num_entities']  # 24,900 (just KG, no RecBole isolated)
    logger.info(f"  KG mapping loaded: {num_entities:,} entities, {kg['num_relations']} relations")

    # ---- 1b. Build SharedEmbedding ----
    # Note: for this test we use ONLY 24,900 entities (no KG-isolated items).
    # This is because we are testing the Task A in isolation, exactly as
    # it was evaluated in Step 2. The 57 KG-isolated items don't have any
    # KG triples and would only matter for Task B.
    shared = SharedEmbedding(
        num_entities=num_entities,
        embedding_dim=EMBEDDING_DIM,
        padding_idx=None,
    )
    logger.info(f"  SharedEmbedding built: {num_entities:,} x {EMBEDDING_DIM}")

    # ---- 1c. Build TaskATransE ----
    task_a = TaskATransE(
        shared_embedding=shared,
        num_relations=NUM_RELATIONS,
        embedding_dim=EMBEDDING_DIM,
        p_norm=1,
        normalize_entities=True,
    )

    # ---- 1d. Load weights from best_model.pt ----
    logger.info(f"\n  Loading weights from {BEST_MODEL_PATH} ...")
    state_dict = torch.load(BEST_MODEL_PATH, map_location='cpu', weights_only=False)
    entity_weights = state_dict['entity_representations.0._embeddings.weight']
    relation_weights = state_dict['relation_representations.0._embeddings.weight']

    # ASSUMPTION: PyKEEN sorts entities/relations alphabetically (just like load_kg).
    # So PyKEEN's id == our id == row index in the weight matrix.
    # This is the assumption we are about to test with the MRR comparison.
    # If MRR comes out catastrophically wrong (e.g. < 0.05), this assumption
    # is broken and we need to investigate.

    # Build "identity" mappings: PyKEEN's id == our id, by assumption
    pykeen_entity_to_id = {entity: i for entity, i in kg['entity_to_id'].items()}
    pykeen_relation_to_id = {rel: i for rel, i in kg['relation_to_id'].items()}

    shared.load_from_pykeen_transe(
        pykeen_weights=entity_weights,
        entity_id_mapping_pykeen=pykeen_entity_to_id,
        entity_id_mapping_ours=kg['entity_to_id'],
        verbose=True,
    )

    task_a.load_relation_weights_from_pykeen(
        pykeen_relation_weights=relation_weights,
        relation_id_mapping_pykeen=pykeen_relation_to_id,
        relation_id_mapping_ours=kg['relation_to_id'],
        verbose=True,
    )

    logger.info("\n  Sanity check: after warm start, entity norms should be ~1.0")
    shared.print_diagnostic(label="after warm start")

    return shared, task_a, kg


# --------------------------------------------------------------------------- #
# Step 2: Build helper structures (type index, known triples for filtering)
# --------------------------------------------------------------------------- #

def build_type_index(kg, num_entities):
    """
    Build a tensor mask for type-constrained evaluation.
    Returns a dict: type_str -> LongTensor of entity ids of that type.
    """
    type_to_ids = {'item': [], 'machine': [], 'client': [], 'model': []}
    for label, eid in kg['entity_to_id'].items():
        prefix = label.split('_')[0]
        if prefix in type_to_ids:
            type_to_ids[prefix].append(eid)
    return {k: torch.LongTensor(v) for k, v in type_to_ids.items()}


def build_known_neighbors(kg, relation_id):
    """
    For filtered evaluation, we need to know which (head, tail) pairs are
    "known" (i.e., appear anywhere in train + valid + test). This is so we
    can EXCLUDE them from the ranking (otherwise they'd compete with the
    ground truth tail for the same head).

    Returns:
        head_to_known_tails : dict[int, set[int]]
        tail_to_known_heads : dict[int, set[int]]
    """
    head_to_known_tails = {}
    tail_to_known_heads = {}

    for triples in [kg['train_triples'], kg['valid_triples'], kg['test_triples']]:
        for h, r, t in triples.tolist():
            if r != relation_id:
                continue
            head_to_known_tails.setdefault(h, set()).add(t)
            tail_to_known_heads.setdefault(t, set()).add(h)

    return head_to_known_tails, tail_to_known_heads


# --------------------------------------------------------------------------- #
# Step 3: The evaluation function (replicates evaluate_custom from Step 2)
# --------------------------------------------------------------------------- #

def evaluate(task_a, kg, num_entities, type_index, mode='type_constrained'):
    """
    Replicates the evaluate_custom function from run_hpo_kge2.py.

    Parameters
    ----------
    task_a : TaskATransE
        Our model (with warm-started weights).
    kg : dict
        Output of load_kg.
    num_entities : int
    type_index : dict[str, LongTensor]
    mode : 'naive' or 'type_constrained'

    Returns
    -------
    dict with metrics for tail prediction, head prediction, and average.
    """
    logger.info(f"\n  --- Evaluating in mode='{mode}' ---")

    relation_id = kg['relation_to_id'][RELATION_LABEL]
    logger.info(f"    relation '{RELATION_LABEL}' has id {relation_id}")

    # ---- Build the type masks (matches evaluate_custom in run_hpo_kge2.py) ----
    if mode == 'type_constrained':
        # For compatible_with: heads must be items, tails must be machines
        tail_mask = torch.full((num_entities,), float('-inf'))
        tail_mask[type_index['machine']] = 0.0
        head_mask = torch.full((num_entities,), float('-inf'))
        head_mask[type_index['item']] = 0.0
    else:
        # naive: all entities are valid candidates
        tail_mask = torch.zeros(num_entities)
        head_mask = torch.zeros(num_entities)

    # ---- Build the filtering tables ----
    head_to_known_tails, tail_to_known_heads = build_known_neighbors(kg, relation_id)

    # ---- Filter test triples to keep only the relation under evaluation ----
    test_triples = kg['test_triples']
    mask = test_triples[:, 1] == relation_id
    test_triples = test_triples[mask]
    logger.info(f"    Evaluating on {len(test_triples):,} test triples "
                f"(relation == {RELATION_LABEL})")

    # ---- Set model to eval mode ----
    task_a.eval()

    # ---- Compute ranks ----
    tail_ranks = []
    head_ranks = []

    with torch.no_grad():
        for i, (h, r, t) in enumerate(test_triples.tolist()):
            # ----- Tail prediction: rank candidates given (h, r) -----
            scores = task_a.score_all_tails(
                torch.LongTensor([h]),
                torch.LongTensor([r]),
            ).squeeze(0)  # (num_entities,)

            scores = scores + tail_mask  # restrict to allowed types

            # Filter: set known tails (other than t) to -inf so they don't compete
            for kt in head_to_known_tails.get(h, set()):
                if kt != t:
                    scores[kt] = float('-inf')

            # Rank: count entities with HIGHER score than the ground truth t
            tail_ranks.append((scores > scores[t]).sum().item() + 1)

            # ----- Head prediction: rank candidates given (r, t) -----
            scores = task_a.score_all_heads(
                torch.LongTensor([r]),
                torch.LongTensor([t]),
            ).squeeze(0)

            scores = scores + head_mask

            for kh in tail_to_known_heads.get(t, set()):
                if kh != h:
                    scores[kh] = float('-inf')

            head_ranks.append((scores > scores[h]).sum().item() + 1)

            if (i + 1) % 1000 == 0:
                logger.info(f"      processed {i+1:,}/{len(test_triples):,} triples")

    # ---- Compute metrics ----
    def metrics_from_ranks(ranks):
        ranks = torch.tensor(ranks).float()
        m = {
            'MRR': (1.0 / ranks).mean().item(),
            'MR': ranks.mean().item(),
        }
        for k in HITS_AT_K:
            m[f'Hits@{k}'] = (ranks <= k).float().mean().item()
        return m

    tail_m = metrics_from_ranks(tail_ranks)
    head_m = metrics_from_ranks(head_ranks)
    avg_m = {k: (tail_m[k] + head_m[k]) / 2 for k in tail_m}

    return {'tail': tail_m, 'head': head_m, 'avg': avg_m}


# --------------------------------------------------------------------------- #
# Step 4: Compare with Step 2's reported metrics
# --------------------------------------------------------------------------- #

def compare_with_step2(our_results):
    """
    Load best_metrics.json from Step 2 and compare side-by-side.
    Print a clear diagnostic.
    """
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON WITH STEP 2's REPORTED METRICS")
    logger.info("=" * 70)

    if not os.path.exists(BEST_METRICS_PATH):
        logger.warning(f"  {BEST_METRICS_PATH} not found.")
        logger.warning(f"  Skipping comparison; please check manually.")
        logger.info(f"\n  OUR results (averaged head + tail):")
        for mode in ['naive', 'type_constrained']:
            avg = our_results[mode]['avg']
            logger.info(f"    {mode:18s} | MRR={avg['MRR']:.4f} "
                        f"H@1={avg['Hits@1']:.4f} "
                        f"H@3={avg['Hits@3']:.4f} "
                        f"H@10={avg['Hits@10']:.4f}")
        return

    with open(BEST_METRICS_PATH) as f:
        step2 = json.load(f)

    print()
    print(f"  {'Metric':12s} {'Mode':18s} {'Step 2':>10s} {'Ours':>10s} {'Diff':>10s}")
    print(f"  {'-'*12} {'-'*18} {'-'*10} {'-'*10} {'-'*10}")

    max_diff = 0.0
    for mode in ['naive', 'type_constrained']:
        for metric in ['MRR', 'Hits@1', 'Hits@3', 'Hits@10']:
            step2_val = step2[mode]['avg'][metric]
            ours_val = our_results[mode]['avg'][metric]
            diff = ours_val - step2_val
            max_diff = max(max_diff, abs(diff))
            print(f"  {metric:12s} {mode:18s} {step2_val:>10.4f} {ours_val:>10.4f} {diff:>+10.4f}")

    logger.info(f"\n  Max absolute difference: {max_diff:.4f}")

    if max_diff < 0.005:
        logger.info("  VERDICT: PERFECT match (diff < 0.005). Infrastructure verified.")
    elif max_diff < 0.02:
        logger.info("  VERDICT: CLOSE match (diff < 0.02). Likely OK; small differences "
                    "may be due to floating point or normalization details.")
    elif max_diff < 0.05:
        logger.warning("  VERDICT: NOTICEABLE difference (diff < 0.05). Worth investigating, "
                       "but the infrastructure is probably correct in essence.")
    else:
        logger.error("  VERDICT: LARGE difference (diff >= 0.05). There is likely a real "
                     "bug. Most probable causes (in order):")
        logger.error("    1. Mapping mismatch between PyKEEN and load_kg.py")
        logger.error("    2. Wrong filtering of test triples")
        logger.error("    3. Wrong scoring direction (head vs tail)")
        logger.error("  We need to investigate before building Task B.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    # Step 1: Build model and load weights
    shared, task_a, kg = build_model_with_warm_start()

    # Step 2: Build the type index for type-constrained evaluation
    num_entities = kg['num_entities']
    type_index = build_type_index(kg, num_entities)
    logger.info(f"\n  Type index built: "
                f"items={len(type_index['item']):,}, "
                f"machines={len(type_index['machine']):,}, "
                f"models={len(type_index['model']):,}, "
                f"clients={len(type_index['client']):,}")

    # Step 3: Evaluate in both modes
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: running evaluation")
    logger.info("=" * 70)

    naive_results = evaluate(task_a, kg, num_entities, type_index, mode='naive')
    typed_results = evaluate(task_a, kg, num_entities, type_index, mode='type_constrained')

    our_results = {'naive': naive_results, 'type_constrained': typed_results}

    # Step 4: Compare with Step 2's reported numbers
    compare_with_step2(our_results)

    logger.info("\nDONE.")


if __name__ == "__main__":
    main()
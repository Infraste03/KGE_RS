"""
Verify Step 2's reported metrics by reloading best_model.pt with PyKEEN
and re-running the evaluate_custom function from run_hpo_kge2.py.

WHY THIS IS IMPORTANT:
    Before debugging our own infrastructure (test_task_a_real_data.py
    showed a difference), we need to confirm that best_metrics.json
    actually corresponds to best_model.pt. If we re-run PyKEEN's eval
    on best_model.pt and it does NOT give MRR=0.2953, then the problem
    is upstream of our infrastructure.

WHAT THIS SCRIPT DOES:
    1. Load TriplesFactory exactly as run_hpo_kge2.py does
    2. Build a fresh TransE model with the same hyperparameters
    3. Load best_model.pt's state_dict into that model
    4. Run the EXACT evaluate_custom function from run_hpo_kge2.py
       (copied verbatim into this file for reproducibility)
    5. Compare the produced numbers with best_metrics.json

EXPECTED OUTCOME:
    If the produced numbers match best_metrics.json -> Step 2 results
    are reproducible, and the problem is in our infrastructure.

    If the produced numbers do NOT match -> Step 2 results were never
    reproducible to begin with, and best_metrics.json may refer to a
    different model checkpoint.

----------------------------------------------------------------------------
Where to put this file:
    B2B_alternate_learning/checks/verify_step2_metrics.py
----------------------------------------------------------------------------
"""

import os
import json
import logging
import torch
import pandas as pd

from pykeen.triples import TriplesFactory
from pykeen.models import TransE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Paths and config (must match run_hpo_kge2.py and best_paramsTransE.json)
# --------------------------------------------------------------------------- #
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KG_PATH = os.path.join(base_dir, "data", "processed", "kg_train.tsv")
TASKA_VALID = os.path.join(base_dir, "data", "processed", "taskA_valid.tsv")
TASKA_TEST = os.path.join(base_dir, "data", "processed", "taskA_test.tsv")

BEST_MODEL_PATH = os.path.join(
    base_dir,
    "results",
    "step2_hpo",
    "TransE",
    "best_model.pt"
)

BEST_METRICS_PATH = os.path.join(
    base_dir,
    "results",
    "step2_hpo",
    "TransE",
    "best_metrics.json"
)

BEST_PARAMS_PATH = os.path.join(
    base_dir,
    "results",
    "step2_hpo",
    "TransE",
    "best_paramsTransE.json"
)

HITS_AT_K = [1, 3, 10]
RELATION_LABEL = 'compatible_with'


# --------------------------------------------------------------------------- #
# evaluate_custom -- copied VERBATIM from run_hpo_kge2.py
# --------------------------------------------------------------------------- #

def evaluate_custom(model, test_factory, train_factory, valid_factory,
                    type_index, mode, relation_label=RELATION_LABEL):
    """Custom evaluation supporting naive and type_constrained modes."""
    device = next(model.parameters()).device
    relation_id = train_factory.relation_to_id[relation_label]

    num_entities = train_factory.num_entities
    if mode == 'type_constrained':
        tail_mask = torch.full((num_entities,), float('-inf'))
        tail_mask[type_index['machine']] = 0.0
        head_mask = torch.full((num_entities,), float('-inf'))
        head_mask[type_index['item']] = 0.0
    else:
        tail_mask = torch.zeros(num_entities)
        head_mask = torch.zeros(num_entities)

    tail_mask = tail_mask.to(device)
    head_mask = head_mask.to(device)

    head_to_known_tails, tail_to_known_heads = {}, {}
    for fac in [train_factory, valid_factory, test_factory]:
        for h, r, t in fac.mapped_triples.tolist():
            if r != relation_id:
                continue
            head_to_known_tails.setdefault(h, set()).add(t)
            tail_to_known_heads.setdefault(t, set()).add(h)

    test_triples = test_factory.mapped_triples
    test_triples = test_triples[test_triples[:, 1] == relation_id]

    model.eval()
    tail_ranks, head_ranks = [], []

    with torch.no_grad():
        for h, r, t in test_triples.tolist():
            scores = model.score_t(torch.tensor([[h, r]], device=device)).squeeze(0)
            scores = scores + tail_mask
            for kt in head_to_known_tails.get(h, set()):
                if kt != t:
                    scores[kt] = float('-inf')
            tail_ranks.append((scores > scores[t]).sum().item() + 1)

            scores = model.score_h(torch.tensor([[r, t]], device=device)).squeeze(0)
            scores = scores + head_mask
            for kh in tail_to_known_heads.get(t, set()):
                if kh != h:
                    scores[kh] = float('-inf')
            head_ranks.append((scores > scores[h]).sum().item() + 1)

    def metrics(ranks):
        ranks = torch.tensor(ranks).float()
        m = {
            'MRR': (1.0 / ranks).mean().item(),
            'MR': ranks.mean().item(),
        }
        for k in HITS_AT_K:
            m[f'Hits@{k}'] = (ranks <= k).float().mean().item()
        return m

    tail_m = metrics(tail_ranks)
    head_m = metrics(head_ranks)
    avg_m = {k: (tail_m[k] + head_m[k]) / 2 for k in tail_m}
    return {'tail': tail_m, 'head': head_m, 'avg': avg_m}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_triples_factories():
    """Same as in run_hpo_kge2.py."""
    train_df = pd.read_csv(KG_PATH, sep='\t')
    valid_df = pd.read_csv(TASKA_VALID, sep='\t')
    test_df = pd.read_csv(TASKA_TEST, sep='\t')

    train_arr = train_df[['head', 'relation', 'tail']].values
    valid_arr = valid_df[['head', 'relation', 'tail']].values
    test_arr = test_df[['head', 'relation', 'tail']].values

    train_factory = TriplesFactory.from_labeled_triples(train_arr)
    valid_factory = TriplesFactory.from_labeled_triples(
        valid_arr,
        entity_to_id=train_factory.entity_to_id,
        relation_to_id=train_factory.relation_to_id,
    )
    test_factory = TriplesFactory.from_labeled_triples(
        test_arr,
        entity_to_id=train_factory.entity_to_id,
        relation_to_id=train_factory.relation_to_id,
    )
    return train_factory, valid_factory, test_factory


def build_type_index(train_factory):
    """Same as in run_hpo_kge2.py."""
    type_to_ids = {'item': [], 'machine': [], 'client': [], 'model': []}
    for label, eid in train_factory.entity_to_id.items():
        prefix = label.split('_')[0]
        if prefix in type_to_ids:
            type_to_ids[prefix].append(eid)
    return {k: torch.LongTensor(v) for k, v in type_to_ids.items()}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    logger.info("=" * 70)
    logger.info("VERIFY STEP 2 METRICS USING PYKEEN NATIVELY")
    logger.info("=" * 70)

    # ---- 1. Load TriplesFactories ----
    logger.info("\n[1/4] Loading TriplesFactories ...")
    train_factory, valid_factory, test_factory = load_triples_factories()
    logger.info(f"  Entities : {train_factory.num_entities:,}")
    logger.info(f"  Relations: {train_factory.num_relations}")
    logger.info(f"  Train    : {train_factory.num_triples:,}")
    logger.info(f"  Valid    : {valid_factory.num_triples:,}")
    logger.info(f"  Test     : {test_factory.num_triples:,}")

    type_index = build_type_index(train_factory)

    # ---- 2. Read best_paramsTransE.json to know embedding_dim ----
    logger.info(f"\n[2/4] Reading {BEST_PARAMS_PATH} ...")
    with open(BEST_PARAMS_PATH) as f:
        best_params = json.load(f)
    embedding_dim = best_params['embedding_dim']
    logger.info(f"  embedding_dim = {embedding_dim}")

    # ---- 3. Build TransE model and load weights ----
    logger.info(f"\n[3/4] Building TransE and loading weights from {BEST_MODEL_PATH} ...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"  Device: {device}")

    model = TransE(
        triples_factory=train_factory,
        embedding_dim=embedding_dim,
    )
    model = model.to(device)

    state_dict = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    # Sanity check: entity norms should be ~1.0 (PyKEEN normalizes them)
    with torch.no_grad():
        ent_w = model.entity_representations[0]._embeddings.weight
        norms = ent_w.norm(dim=1)
        logger.info(f"  Entity norms: mean={norms.mean().item():.4f}, "
                    f"std={norms.std().item():.4f}")

    # ---- 4. Run evaluate_custom in both modes ----
    logger.info(f"\n[4/4] Running evaluate_custom in both modes ...")
    logger.info(f"  This will take a few minutes (same as test_task_a_real_data.py)")

    logger.info("\n  --- mode=naive ---")
    naive_results = evaluate_custom(
        model, test_factory, train_factory, valid_factory,
        type_index, mode='naive',
    )

    logger.info("\n  --- mode=type_constrained ---")
    typed_results = evaluate_custom(
        model, test_factory, train_factory, valid_factory,
        type_index, mode='type_constrained',
    )

    # ---- 5. Compare with best_metrics.json ----
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON WITH best_metrics.json")
    logger.info("=" * 70)

    with open(BEST_METRICS_PATH) as f:
        reported = json.load(f)

    print()
    print(f"  {'Metric':12s} {'Mode':18s} {'reported':>10s} {'reproduced':>12s} {'Diff':>10s}")
    print(f"  {'-'*12} {'-'*18} {'-'*10} {'-'*12} {'-'*10}")

    max_diff = 0.0
    our_results = {'naive': naive_results, 'type_constrained': typed_results}

    for mode in ['naive', 'type_constrained']:
        for metric in ['MRR', 'Hits@1', 'Hits@3', 'Hits@10']:
            rep_val = reported[mode]['avg'][metric]
            ours_val = our_results[mode]['avg'][metric]
            diff = ours_val - rep_val
            max_diff = max(max_diff, abs(diff))
            print(f"  {metric:12s} {mode:18s} {rep_val:>10.4f} {ours_val:>12.4f} {diff:>+10.4f}")

    logger.info(f"\n  Max absolute difference: {max_diff:.4f}")

    if max_diff < 0.005:
        logger.info("  VERDICT: best_model.pt IS the model that produced best_metrics.json.")
        logger.info("  -> The discrepancy in test_task_a_real_data.py is due to a bug")
        logger.info("     in OUR infrastructure (TaskATransE / SharedEmbedding).")
    elif max_diff < 0.05:
        logger.info("  VERDICT: small reproducibility gap. best_metrics.json may have been")
        logger.info("  produced from a slightly different model state, but the differences")
        logger.info("  are within typical floating-point/seed variability.")
    else:
        logger.warning("  VERDICT: best_metrics.json does NOT correspond to best_model.pt!")
        logger.warning("  The 'true' baseline for best_model.pt is the 'reproduced' column.")
        logger.warning("  Compare test_task_a_real_data.py against THIS column, not against")
        logger.warning("  best_metrics.json.")


if __name__ == "__main__":
    main()
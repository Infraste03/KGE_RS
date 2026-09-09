"""
test_task_a_real_data_fashion.py
==================================

Verify that TaskATransE fashion exactly reproduces the metrics
by PyKEEN on the test set (MRR=0.1057, H@1=0.0691, H@3=0.1183, H@10=0.1755).

This is the most important check before proceeding with training:
If our infrastructure doesn't replicate PyKEEN, the warm start weights
would be loaded incorrectly and training would start from the wrong bases.

Fashion note: compatible_with is item->item, so type_constrained
and naive produce the same result (verified in verify_task_a_type_index_fashion).


"""

import os
import sys
import logging
import torch
import pandas as pd

THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
MODELS_DIR = os.path.join(PARENT_DIR, "models")
BASE_DIR   = os.path.abspath(os.path.join(PARENT_DIR, ".."))


if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from load_kg_fashion import load_kg_fashion
from shared_embedding_fashion import SharedEmbedding
from task_a_transe_fashion import TaskATransE
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
KG_TRAIN    = os.path.join(BASE_DIR, "data", "processed", "kg_train.tsv")
KG_VALID    = os.path.join(BASE_DIR, "data", "processed", "taskA_valid.tsv")
KG_TEST     = os.path.join(BASE_DIR, "data", "processed", "taskA_test.tsv")
TRANSE_PATH = os.path.join(BASE_DIR, "results", "hpo_transe", "best_model.pt")

# Metriche attese dal log HPO (avg head+tail, type_constrained)
EXPECTED = {
    'MRR'    : 0.1057,
    'Hits@1' : 0.0691,
    'Hits@3' : 0.1183,
    'Hits@10': 0.1755,
}

EMBEDDING_DIM   = 64
NUM_RELATIONS   = 3
RELATION_LABEL  = 'compatible_with'
HITS_AT_K       = [1, 3, 10]
PADDING_IDX     = 182_983
NUM_ROWS        = 182_984




def build_model():
    logger.info("=" * 70)
    logger.info("STEP 1: Model construction with warm start from real weights")
    logger.info("=" * 70)

    # KG
    kg = load_kg_fashion(KG_TRAIN, KG_VALID, KG_TEST, verbose=False)
    logger.info(f"  KG: {kg['num_entities']:,} entità, {kg['num_relations']} relazioni")

    # PyKEEN mapping
    train_df = pd.read_csv(KG_TRAIN, sep='\t')
    train_factory = TriplesFactory.from_labeled_triples(
        train_df[['head', 'relation', 'tail']].values
    )
    pykeen_entity_to_id   = dict(train_factory.entity_to_id)
    pykeen_relation_to_id = dict(train_factory.relation_to_id)
    logger.info(f"  PyKEEN mapping: {len(pykeen_entity_to_id):,} entities")

    # SharedEmbedding
    shared = SharedEmbedding(
        num_entities=NUM_ROWS,
        embedding_dim=EMBEDDING_DIM,
        padding_idx=PADDING_IDX,
    )

    # TaskATransE
    task_a = TaskATransE(
        shared_embedding=shared,
        num_relations=NUM_RELATIONS,
        embedding_dim=EMBEDDING_DIM,
        p_norm=1,
        normalize_entities=True,
    )



    logger.info(f"\n  Loading weights from {TRANSE_PATH}...")
    state_dict      = torch.load(TRANSE_PATH, map_location='cpu', weights_only=False)
    entity_weights  = state_dict['entity_representations.0._embeddings.weight']
    relation_weights = state_dict['relation_representations.0._embeddings.weight']

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

    shared.print_diagnostic(label="dopo warm start")

    return shared, task_a, kg, pykeen_entity_to_id



def build_type_index(kg: dict) -> dict:
    """
    Fashion: item, category, brand.
    for compatible_with (item->item): head and tail are both items.
    """
    type_to_ids = {'item': [], 'category': [], 'brand': []}
    for label, eid in kg['entity_to_id'].items():
        prefix = label.split('_')[0]
        if prefix in type_to_ids:
            type_to_ids[prefix].append(eid)
    return {k: torch.LongTensor(v) for k, v in type_to_ids.items()}


def build_known_neighbors(kg: dict, relation_id: int):
    head_to_known_tails = {}
    tail_to_known_heads = {}
    for triples in [kg['train_triples'], kg['valid_triples'], kg['test_triples']]:
        for h, r, t in triples.tolist():
            if r != relation_id:
                continue
            head_to_known_tails.setdefault(h, set()).add(t)
            tail_to_known_heads.setdefault(t, set()).add(h)
    return head_to_known_tails, tail_to_known_heads



def evaluate(task_a, kg, type_index):
    """
    Exact replica of evaluate_custom used in TransE fashion HPO.

    Fashion: compatible_with is item->item, so type_constrained
    apply mask item on BOTH head and tail - equivalent to naive
    on the sub-population of items. The two ways coincide because
    all relevant candidate entities are items.
    """
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: valutazione su test set")
    logger.info("=" * 70)

    num_entities = kg['num_entities']  # 182,983
    relation_id  = kg['relation_to_id'][RELATION_LABEL]
    logger.info(f"  Relazione '{RELATION_LABEL}' ha id={relation_id}")


    tail_mask = torch.full((num_entities,), float('-inf'))
    tail_mask[type_index['item']] = 0.0
    head_mask = torch.full((num_entities,), float('-inf'))
    head_mask[type_index['item']] = 0.0

    head_to_known_tails, tail_to_known_heads = build_known_neighbors(kg, relation_id)


    test_triples = kg['test_triples']
    test_triples = test_triples[test_triples[:, 1] == relation_id]
    logger.info(f"  Triple compatible_with in the test: {len(test_triples):,}")

    task_a.eval()
    tail_ranks = []
    head_ranks = []

    with torch.no_grad():
        for i, (h, r, t) in enumerate(test_triples.tolist()):

            scores = task_a.score_all_tails(
                torch.LongTensor([h]),
                torch.LongTensor([r]),
            ).squeeze(0)   # (num_entities,)

            scores = scores[:num_entities]

            scores = scores + tail_mask

            for kt in head_to_known_tails.get(h, set()):
                if kt != t:
                    scores[kt] = float('-inf')

            tail_ranks.append((scores > scores[t]).sum().item() + 1)

            scores = task_a.score_all_heads(
                torch.LongTensor([r]),
                torch.LongTensor([t]),
            ).squeeze(0)

            scores = scores[:num_entities]
            scores = scores + head_mask

            for kh in tail_to_known_heads.get(t, set()):
                if kh != h:
                    scores[kh] = float('-inf')

            head_ranks.append((scores > scores[h]).sum().item() + 1)

            if (i + 1) % 500 == 0:
                logger.info(f"    {i+1:,}/{len(test_triples):,} processed triples")

    def metrics(ranks):
        r = torch.tensor(ranks, dtype=torch.float)
        m = {'MRR': (1.0 / r).mean().item(), 'MR': r.mean().item()}
        for k in HITS_AT_K:
            m[f'Hits@{k}'] = (r <= k).float().mean().item()
        return m

    tail_m = metrics(tail_ranks)
    head_m = metrics(head_ranks)
    avg_m  = {k: (tail_m[k] + head_m[k]) / 2 for k in tail_m}
    return tail_m, head_m, avg_m



def compare(tail_m, head_m, avg_m):
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON WITH EXPECTED METRICS FROM HPO LOG")
    logger.info("=" * 70)

    print(f"\n  {'Metric':10s} {'Tail':>10s} {'Head':>10s} {'Avg':>10s} "
          f"{'expected':>10s} {'Diff':>10s}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    max_diff = 0.0
    for metric in ['MRR', 'Hits@1', 'Hits@3', 'Hits@10']:
        atteso = EXPECTED[metric]
        diff   = avg_m[metric] - atteso
        max_diff = max(max_diff, abs(diff))
        print(f"  {metric:10s} {tail_m[metric]:>10.4f} {head_m[metric]:>10.4f} "
              f"{avg_m[metric]:>10.4f} {atteso:>10.4f} {diff:>+10.4f}")

    logger.info(f"\n  Max absolute difference: {max_diff:.4f}")

    if max_diff < 0.005:
        logger.info("  VERDICT: PASS - PyKEEN replica infrastructure exactly.")
        logger.info("  Warm start weights are valid. Proceed with Task B.")
    elif max_diff < 0.02:
        logger.info("  VERDICT: PASS with small numerical variation (floating point).")
    else:
        logger.error(f"  VERDICT: FAIL — diff {max_diff:.4f} is too large.")
        logger.error("  Probable causes (in order):")
        logger.error("    1. Mismatch nel mapping PyKEEN vs load_kg_fashion")
        logger.error("    2. wrong p_norm (try p_norm=2 instead of 1)")
        logger.error("    3. normalize_entities does not match PyKEEN")
        logger.error("    4. Incorrect filter of test triples")



def main():
    shared, task_a, kg, pykeen_mapping = build_model()

    type_index = build_type_index(kg)
    logger.info(f"\n  Type index: item={len(type_index['item']):,}, "
                f"category={len(type_index['category'])}, "
                f"brand={len(type_index['brand']):,}")

    tail_m, head_m, avg_m = evaluate(task_a, kg, type_index)
    compare(tail_m, head_m, avg_m)

    logger.info("\nDONE.")


if __name__ == "__main__":
    main()
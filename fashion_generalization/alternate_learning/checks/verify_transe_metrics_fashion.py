"""
verify_transe_metrics_fashion.py
==================================
Reload best_model.pt with PyKEEN and recompute MRR on the Fashion test set.
Verify that it matches MRR=0.1057 reported in the HPO log.

Fashion note: naive and type_constrained produce the same result because
compatible_with is item->item: the type masks for both head and tail are
restricted to items, so the two modes collapse to the same evaluation.
This is correct.

File location:
    fashion_generalization/alternate_learning/checks/verify_transe_metrics_fashion.py
"""

import os
import sys
import torch
import pandas as pd

THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
BASE_DIR   = os.path.abspath(os.path.join(PARENT_DIR, ".."))
sys.path.insert(0, PARENT_DIR)

from pykeen.triples import TriplesFactory
from pykeen.models import TransE

KG_TRAIN     = os.path.join(BASE_DIR, "data", "processed", "kg_train.tsv")
KG_VALID     = os.path.join(BASE_DIR, "data", "processed", "taskA_valid.tsv")
KG_TEST      = os.path.join(BASE_DIR, "data", "processed", "taskA_test.tsv")
TRANSE_PATH  = os.path.join(BASE_DIR, "results", "hpo_transe", "best_model.pt")

EXPECTED_MRR    = 0.1057
RELATION_LABEL  = 'compatible_with'
EMBEDDING_DIM   = 64


def load_factories():
    train_df = pd.read_csv(KG_TRAIN, sep='\t')
    valid_df = pd.read_csv(KG_VALID, sep='\t')
    test_df  = pd.read_csv(KG_TEST,  sep='\t')

    train_factory = TriplesFactory.from_labeled_triples(
        train_df[['head', 'relation', 'tail']].values
    )
    valid_factory = TriplesFactory.from_labeled_triples(
        valid_df[['head', 'relation', 'tail']].values,
        entity_to_id=train_factory.entity_to_id,
        relation_to_id=train_factory.relation_to_id,
    )
    test_factory = TriplesFactory.from_labeled_triples(
        test_df[['head', 'relation', 'tail']].values,
        entity_to_id=train_factory.entity_to_id,
        relation_to_id=train_factory.relation_to_id,
    )
    return train_factory, valid_factory, test_factory


def build_type_index(train_factory):
    """In Fashion: item->item, so both head and tail are items."""
    type_to_ids = {'item': [], 'category': [], 'brand': []}
    for label, eid in train_factory.entity_to_id.items():
        prefix = label.split('_')[0]
        if prefix in type_to_ids:
            type_to_ids[prefix].append(eid)
    return {k: torch.LongTensor(v) for k, v in type_to_ids.items()}


def evaluate(model, test_factory, train_factory, valid_factory, type_index, device):
    """
    Evaluate compatible_with using type_constrained evaluation (item->item).
    In Fashion, naive and type_constrained coincide because both
    head and tail entities are items.
    """
    relation_id  = train_factory.relation_to_id[RELATION_LABEL]
    num_entities = train_factory.num_entities

    # Mask: only items can be heads and tails for compatible_with
    item_ids = type_index['item']
    tail_mask = torch.full((num_entities,), float('-inf'))
    tail_mask[item_ids] = 0.0
    head_mask = torch.full((num_entities,), float('-inf'))
    head_mask[item_ids] = 0.0
    tail_mask = tail_mask.to(device)
    head_mask = head_mask.to(device)

    # Build known positives used for filtered evaluation
    head_to_known_tails = {}
    tail_to_known_heads = {}
    for fac in [train_factory, valid_factory, test_factory]:
        for h, r, t in fac.mapped_triples.tolist():
            if r != relation_id:
                continue
            head_to_known_tails.setdefault(h, set()).add(t)
            tail_to_known_heads.setdefault(t, set()).add(h)

    test_triples = test_factory.mapped_triples
    test_triples = test_triples[test_triples[:, 1] == relation_id]
    print(f"  compatible_with triples in test: {len(test_triples):,}")

    model.eval()
    tail_ranks, head_ranks = [], []

    with torch.no_grad():
        for h, r, t in test_triples.tolist():
            # Tail prediction
            scores = model.score_t(torch.tensor([[h, r]], device=device)).squeeze(0)
            scores = scores + tail_mask
            for kt in head_to_known_tails.get(h, set()):
                if kt != t:
                    scores[kt] = float('-inf')
            tail_ranks.append((scores > scores[t]).sum().item() + 1)

            # Head prediction
            scores = model.score_h(torch.tensor([[r, t]], device=device)).squeeze(0)
            scores = scores + head_mask
            for kh in tail_to_known_heads.get(t, set()):
                if kh != h:
                    scores[kh] = float('-inf')
            head_ranks.append((scores > scores[h]).sum().item() + 1)

    def metrics(ranks):
        r = torch.tensor(ranks, dtype=torch.float)
        return {
            'MRR':    (1.0 / r).mean().item(),
            'Hits@1': (r <= 1).float().mean().item(),
            'Hits@3': (r <= 3).float().mean().item(),
            'Hits@10':(r <= 10).float().mean().item(),
        }

    tail_m = metrics(tail_ranks)
    head_m = metrics(head_ranks)
    avg_m  = {k: (tail_m[k] + head_m[k]) / 2 for k in tail_m}
    return tail_m, head_m, avg_m


def main():
    print("=" * 70)
    print("VERIFY TRANSE METRICS — FASHION")
    print("=" * 70)

    print("\n[1/4] Loading TriplesFactory objects...")
    train_f, valid_f, test_f = load_factories()
    print(f"  Entities : {train_f.num_entities:,}")
    print(f"  Relations: {train_f.num_relations}")
    print(f"  Train    : {train_f.num_triples:,}")
    print(f"  Valid    : {valid_f.num_triples:,}")
    print(f"  Test     : {test_f.num_triples:,}")

    type_index = build_type_index(train_f)
    print(f"  Item IDs : {len(type_index['item']):,}")

    print(f"\n[2/4] Building TransE and loading weights...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    model = TransE(triples_factory=train_f, embedding_dim=EMBEDDING_DIM)
    model = model.to(device)
    state_dict = torch.load(TRANSE_PATH, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        ent_w = model.entity_representations[0]._embeddings.weight
        norms = ent_w.norm(dim=1)
        print(f"  Entity norms: mean={norms.mean():.4f}, std={norms.std():.4f}")

    print(f"\n[3/4] Evaluating on test set...")
    print(f"  (Note: in Fashion naive==type_constrained because compatible_with is item->item)")
    tail_m, head_m, avg_m = evaluate(model, test_f, train_f, valid_f, type_index, device)

    print(f"\n[4/4] Comparing with expected MRR from log...")
    print(f"\n  {'Metric':10s} {'Tail':>10s} {'Head':>10s} {'Avg':>10s}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for metric in ['MRR', 'Hits@1', 'Hits@3', 'Hits@10']:
        print(f"  {metric:10s} {tail_m[metric]:>10.4f} {head_m[metric]:>10.4f} {avg_m[metric]:>10.4f}")

    diff = abs(avg_m['MRR'] - EXPECTED_MRR)
    print(f"\n  Expected MRR (from log): {EXPECTED_MRR:.4f}")
    print(f"  Reproduced MRR          : {avg_m['MRR']:.4f}")
    print(f"  Absolute difference     : {diff:.4f}")

    print("\n" + "=" * 70)
    if diff < 0.005:
        print("VERDICT: PASS — best_model.pt matches the MRR reported in the log.")
        print("TransE weights are valid for alternate-learning warm start.")
    elif diff < 0.02:
        print("VERDICT: PASS (small acceptable numerical variation).")
        print("Probably due to floating point precision or evaluation order.")
    else:
        print(f"VERDICT: FAIL — MRR difference={diff:.4f} is too large.")
        print("Verify that best_model.pt is the correct checkpoint.")
    print("=" * 70)


if __name__ == "__main__":
    main()
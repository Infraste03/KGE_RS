"""
verify_task_a_type_index_fashion.py
=====================================
Verify that the type_index for Task A is correctly constructed
for the Fashion KG (types: item, category, brand).

Critical difference compared with B2B:
  - B2B: type_constrained uses 'machine' as the tail type for compatible_with
  - Fashion: compatible_with is item->item, therefore both head and tail
    must be filtered using 'item'

File location:
    fashion_generalization/alternate_learning/checks/verify_task_a_type_index_fashion.py
"""

import os
import sys
import torch

THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
BASE_DIR   = os.path.abspath(os.path.join(PARENT_DIR, ".."))
sys.path.insert(0, PARENT_DIR)

from load_kg_fashion import load_kg_fashion

KG_TRAIN = os.path.join(BASE_DIR, "data", "processed", "kg_train.tsv")
KG_VALID = os.path.join(BASE_DIR, "data", "processed", "taskA_valid.tsv")
KG_TEST  = os.path.join(BASE_DIR, "data", "processed", "taskA_test.tsv")


def build_type_index_fashion(kg: dict) -> dict:
    """
    Build the type_index for the Fashion KG.
    Types: item, category, brand.

    For the type_constrained evaluation of compatible_with (item->item):
      - head candidates: type_index['item']
      - tail candidates: type_index['item']
    """
    type_to_ids = {'item': [], 'category': [], 'brand': []}
    for label, eid in kg['entity_to_id'].items():
        prefix = label.split('_')[0]
        if prefix in type_to_ids:
            type_to_ids[prefix].append(eid)
        else:
            raise ValueError(
                f"Unknown prefix '{prefix}' for entity '{label}'. "
                f"Expected: {list(type_to_ids.keys())}"
            )
    return {k: torch.LongTensor(sorted(v)) for k, v in type_to_ids.items()}


def main():
    print("=" * 70)
    print("VERIFY TASK A TYPE INDEX — FASHION")
    print("=" * 70)

    print("\n[1/3] Loading Fashion KG...")
    kg = load_kg_fashion(KG_TRAIN, KG_VALID, KG_TEST, verbose=False)
    print(f"  Total entities: {kg['num_entities']:,}")

    print("\n[2/3] Building type_index...")
    type_index = build_type_index_fashion(kg)

    for tipo, tensor in type_index.items():
        print(f"  type_index['{tipo}']: shape={tuple(tensor.shape)}, "
              f"dtype={tensor.dtype}, "
              f"range=[{tensor.min().item()}, {tensor.max().item()}]")

    print("\n[3/3] Assertions...")

    # Each type must be a non-empty 1D LongTensor
    for tipo in ['item', 'category', 'brand']:
        t = type_index[tipo]
        assert isinstance(t, torch.Tensor), f"type_index['{tipo}'] is not a Tensor"
        assert t.ndim == 1,                 f"type_index['{tipo}'] is not 1D"
        assert t.dtype == torch.long,       f"type_index['{tipo}'] is not a LongTensor"
        assert t.numel() > 0,               f"type_index['{tipo}'] is empty"
        print(f"  PASS: '{tipo}' -> {t.numel():,} entities")

    # The three sets must be disjoint and cover all entities
    all_ids = torch.cat([type_index[t] for t in ['item', 'category', 'brand']])
    assert all_ids.numel() == kg['num_entities'], \
        f"type_index does not cover all entities: {all_ids.numel()} vs {kg['num_entities']}"
    assert len(all_ids.tolist()) == len(set(all_ids.tolist())), \
        "Duplicate IDs found across entity types!"
    print(f"  PASS: the three types cover all {kg['num_entities']:,} entities without duplicates")

    # Verify that compatible_with uses only items as heads and tails
    relation_id_compat = kg['relation_to_id']['compatible_with']
    train_triples = kg['train_triples']
    compat_triples = train_triples[train_triples[:, 1] == relation_id_compat]

    item_ids_set = set(type_index['item'].tolist())
    heads = set(compat_triples[:, 0].tolist())
    tails = set(compat_triples[:, 2].tolist())

    non_item_heads = heads - item_ids_set
    non_item_tails = tails - item_ids_set

    assert len(non_item_heads) == 0, \
        f"compatible_with has {len(non_item_heads)} non-item heads: {list(non_item_heads)[:5]}"
    assert len(non_item_tails) == 0, \
        f"compatible_with has {len(non_item_tails)} non-item tails: {list(non_item_tails)[:5]}"
    print(f"  PASS: compatible_with uses only items as heads and tails")
    print(f"        -> type_constrained and naive are equivalent for this relation")

    print("\n" + "=" * 70)
    print("TASK A TYPE INDEX CHECK PASSED")
    print("=" * 70)
    print()
    print("Summary for warm start and training:")
    print(f"  SharedEmbedding will have {kg['num_entities']:,} rows ({EXPECTED_EMBED_DIM} dim)")
    print(f"  Task A (TransE): trains on {len(compat_triples):,} compatible_with triples")
    print(f"  Evaluation type mask: item IDs for both heads and tails")

EXPECTED_EMBED_DIM = 64

if __name__ == "__main__":
    main()
import os
import sys
import torch

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
from pykeen.triples import TriplesFactory
import pandas as pd

KG_TRAIN    = os.path.join(BASE_DIR, "data", "processed", "kg_train.tsv")
KG_VALID    = os.path.join(BASE_DIR, "data", "processed", "taskA_valid.tsv")
KG_TEST     = os.path.join(BASE_DIR, "data", "processed", "taskA_test.tsv")
TRANSE_PATH = os.path.join(BASE_DIR, "results", "hpo_transe", "best_model.pt")

EXPECTED_NUM_ENTITIES = 182_983
EXPECTED_DIM          = 64
PADDING_IDX           = EXPECTED_NUM_ENTITIES       # = 182,983
NUM_ROWS              = EXPECTED_NUM_ENTITIES + 1   # = 182,984


def load_pykeen_mapping() -> dict:

    train_df = pd.read_csv(KG_TRAIN, sep='\t')
    valid_df = pd.read_csv(KG_VALID, sep='\t')
    test_df  = pd.read_csv(KG_TEST,  sep='\t')

    train_factory = TriplesFactory.from_labeled_triples(
        train_df[['head', 'relation', 'tail']].values
    )

    _ = TriplesFactory.from_labeled_triples(
        valid_df[['head', 'relation', 'tail']].values,
        entity_to_id=train_factory.entity_to_id,
        relation_to_id=train_factory.relation_to_id,
    )
    _ = TriplesFactory.from_labeled_triples(
        test_df[['head', 'relation', 'tail']].values,
        entity_to_id=train_factory.entity_to_id,
        relation_to_id=train_factory.relation_to_id,
    )
    return dict(train_factory.entity_to_id)


def main():
    print("=" * 70)
    print("VERIFY SHARED EMBEDDING WARM START — FASHION (real weights)")
    print("=" * 70)

    # ---- 1. Load KG and TransE weights ----
    print("\n[1/5] Loading KG and TransE checkpoint...")
    kg = load_kg_fashion(KG_TRAIN, KG_VALID, KG_TEST, verbose=False)

    assert kg['num_entities'] == EXPECTED_NUM_ENTITIES, \
        f"Expected {EXPECTED_NUM_ENTITIES} KG entities, found {kg['num_entities']}"

    state_dict    = torch.load(TRANSE_PATH, map_location='cpu', weights_only=False)
    pykeen_weights = state_dict['entity_representations.0._embeddings.weight']
    print(f"  KG entities      : {kg['num_entities']:,}")
    print(f"  TransE weights   : {tuple(pykeen_weights.shape)}")
    assert pykeen_weights.shape == (EXPECTED_NUM_ENTITIES, EXPECTED_DIM), \
        f"Expected shape ({EXPECTED_NUM_ENTITIES}, {EXPECTED_DIM}), " \
        f"found {tuple(pykeen_weights.shape)}"

    # ---- 2. Rebuild PyKEEN mapping ----
    print("\n[2/5] Rebuilding PyKEEN mapping (TriplesFactory)...")
    pykeen_mapping = load_pykeen_mapping()
    print(f"  Entities in PyKEEN mapping: {len(pykeen_mapping):,}")
    assert len(pykeen_mapping) == EXPECTED_NUM_ENTITIES, \
        f"PyKEEN mapping: expected {EXPECTED_NUM_ENTITIES} entities, found {len(pykeen_mapping)}"

    # ---- 3. Build SharedEmbedding and load weights ----
    print("\n[3/5] Building SharedEmbedding and applying warm start...")
    shared = SharedEmbedding(
        num_entities=NUM_ROWS,
        embedding_dim=EXPECTED_DIM,
        padding_idx=PADDING_IDX,
    )

    norm_before = shared.embedding.weight.norm(p=2, dim=1).mean().item()

    shared.load_from_pykeen_transe(
        pykeen_weights=pykeen_weights,
        entity_id_mapping_pykeen=pykeen_mapping,
        entity_id_mapping_ours=kg['entity_to_id'],
        verbose=True,
    )

    norm_after = shared.embedding.weight[:EXPECTED_NUM_ENTITIES].norm(p=2, dim=1).mean().item()
    print(f"  L2 norm before warm start : {norm_before:.4f}")
    print(f"  L2 norm after warm start  : {norm_after:.4f}")

    # ---- 4. Assertions on loaded weights ----
    print("\n[4/5] Assertions...")

    # 4a. Norm ~1.0 for all KG entities (PyKEEN normalizes them)
    norms = shared.embedding.weight[:EXPECTED_NUM_ENTITIES].norm(p=2, dim=1)
    assert abs(norms.mean().item() - 1.0) < 0.01, \
        f"Expected mean L2 norm ~1.0, found {norms.mean().item():.4f}. " \
        f"TransE weights do not appear to be normalized."
    assert norms.std().item() < 0.01, \
        f"Expected L2 norm std ~0.0, found {norms.std().item():.4f}."
    print(f"  PASS: mean L2 norm = {norms.mean().item():.4f} (expected ~1.0)")
    print(f"  PASS: L2 norm std  = {norms.std().item():.6f} (expected ~0.0)")

    # 4b. Padding row is zero
    pad_norm = shared.embedding.weight[PADDING_IDX].norm(p=2).item()
    assert pad_norm == 0.0, \
        f"Padding row (idx={PADDING_IDX}) is not zero: norm={pad_norm}"
    print(f"  PASS: padding row (idx={PADDING_IDX}) is zero")

    # 4c. No NaN, no Inf
    assert not torch.isnan(shared.embedding.weight).any(), "NaN values found!"
    assert not torch.isinf(shared.embedding.weight).any(), "Inf values found!"
    print(f"  PASS: no NaN or Inf values")

    # ---- 5. Verify exact copy on a sample ----
    print("\n[5/5] Verifying exact copy on a sample of entities...")

    # Sample one entity for each type
    sample_entities = {
        'brand'   : next(e for e in kg['entity_to_id'] if e.startswith('brand_')),
        'category': next(e for e in kg['entity_to_id'] if e.startswith('category_')),
        'item'    : next(e for e in kg['entity_to_id'] if e.startswith('item_')),
    }

    for tipo, entity_str in sample_entities.items():
        our_id    = kg['entity_to_id'][entity_str]
        pykeen_id = pykeen_mapping[entity_str]

        weight_in_shared   = shared.embedding.weight[our_id].detach().cpu()
        weight_in_pykeen   = pykeen_weights[pykeen_id].cpu()

        diff = (weight_in_shared - weight_in_pykeen).abs().max().item()
        assert diff == 0.0, \
            f"Incorrect copy for {entity_str} (type={tipo}): " \
            f"our_id={our_id}, pykeen_id={pykeen_id}, diff={diff}"
        print(f"  PASS: {tipo:10s} '{entity_str}' -> "
              f"our_id={our_id}, pykeen_id={pykeen_id}, diff={diff:.2e}")

    # Print final diagnostic
    shared.print_diagnostic(label="after real warm start")

    print("\n" + "=" * 70)
    print("VERIFY SHARED EMBEDDING WARM START PASSED")
    print("=" * 70)
    print()
    print("SharedEmbedding with real TransE weights is ready for alternate learning.")
    print("Next step: task_a_transe_fashion.py")


if __name__ == "__main__":
    main()
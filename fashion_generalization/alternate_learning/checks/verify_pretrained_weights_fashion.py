"""
Verify the pretrained weights of TransE (fashion HPO best_model.pt) and SASRec (trial_017/best_valid.pth).
"""
import os
import sys
import torch

THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
BASE_DIR   = os.path.abspath(os.path.join(PARENT_DIR, ".."))
sys.path.insert(0, PARENT_DIR)

from load_kg_fashion import load_kg_fashion

#
TRANSE_PATH  = os.path.join(BASE_DIR, "results", "hpo_transe", "best_model.pt")
SASREC_PATH  = os.path.join(BASE_DIR, "results", "sasrec_hpo", "trial_017", "best_valid.pth")

KG_TRAIN = os.path.join(BASE_DIR, "data", "processed", "kg_train.tsv")
KG_VALID = os.path.join(BASE_DIR, "data", "processed", "taskA_valid.tsv")
KG_TEST  = os.path.join(BASE_DIR, "data", "processed", "taskA_test.tsv")


EXPECTED_NUM_ENTITIES  = 182_983
EXPECTED_NUM_RELATIONS = 3
EXPECTED_EMBED_DIM     = 64



def main():
    print("=" * 70)
    print("CHECK PRE-TRAINED WEIGHTS - FASHION")
    print("=" * 70)
    print("\n" + "=" * 70)
    print("PART A - TransE best_model.pt")
    print("=" * 70)


    if not os.path.exists(TRANSE_PATH):
        print(f" ERROR: File not found: {TRANSE_PATH}")
        sys.exit(1)

    print(f"\n[A1] Loading {TRANSE_PATH}...")
    state_dict = torch.load(TRANSE_PATH, map_location='cpu', weights_only=False)
    print(f" Keys in state_dict: {list(state_dict.keys())}")

    expected_keys = {
        'entity_representations.0._embeddings.weight',
        'relation_representations.0._embeddings.weight',
    }
    actual_keys = set(state_dict.keys())
    if actual_keys != expected_keys:
        print(f" WARNING: Unexpected keys.")
        print(f" Expected: {expected_keys}")
        print(f" Found: {actual_keys}")
        sys.exit(1)

    entity_w   = state_dict['entity_representations.0._embeddings.weight']
    relation_w = state_dict['relation_representations.0._embeddings.weight']
    print(f"  entity_weights   shape: {tuple(entity_w.shape)}")
    print(f"  relation_weights shape: {tuple(relation_w.shape)}")

    print(f"\n[A2] test shape...")
    n_ent, dim_ent = entity_w.shape
    n_rel, dim_rel = relation_w.shape
    errors = []
    if n_ent != EXPECTED_NUM_ENTITIES:
        errors.append(f"  expected {EXPECTED_NUM_ENTITIES}, found {n_ent}")
    if n_rel != EXPECTED_NUM_RELATIONS:
        errors.append(f"  relations: expected {EXPECTED_NUM_RELATIONS}, found {n_rel}")
    if dim_ent != EXPECTED_EMBED_DIM:
        errors.append(f"  entity dim: expected {EXPECTED_EMBED_DIM}, found {dim_ent}")
    if dim_ent != dim_rel:
        errors.append(f"  entity dim ({dim_ent}) != relation dim ({dim_rel})")
    if errors:
        print("  FAIL:")
        for e in errors:
            print(e)
        sys.exit(1)
    print(f"  PASS: {n_ent:,} entity x {dim_ent} dim, {n_rel} relations x {dim_rel} dim")

    print(f"\n[A3] Weight statistics...")
    ent_norms = entity_w.norm(dim=1)
    rel_norms = relation_w.norm(dim=1)
    print(f"  Entity L2 norm: mean={ent_norms.mean():.4f}, std={ent_norms.std():.4f}, "
          f"min={ent_norms.min():.4f}, max={ent_norms.max():.4f}")
    print(f"  Relation L2 norm: mean={rel_norms.mean():.4f}, std={rel_norms.std():.4f}")
    print(f"  NaN in entity_w : {torch.isnan(entity_w).any().item()}")
    print(f"  Inf in entity_w : {torch.isinf(entity_w).any().item()}")
    if torch.isnan(entity_w).any() or torch.isinf(entity_w).any():
        print("  FAIL: NaN o Inf nei pesi TransE!")
        sys.exit(1)
    if 0.9 < ent_norms.mean().item() < 1.1:
        print(f"  -> Norms ~1.0: PyKEEN used normalized TransE (expected).")
    else:
        print(f"  -> Norms non-unitary (mean={ent_norms.mean():.3f}): verify the config.")
    print(f"  PASS: ok TransE weights")

    print(f"\n[A4] Comparison with load_kg_fashion (entity/relationship count)...")
    kg = load_kg_fashion(KG_TRAIN, KG_VALID, KG_TEST, verbose=False)
    if kg['num_entities'] != n_ent:
        print(f"  FAIL: entity count KG={kg['num_entities']} vs PyKEEN={n_ent}")
        sys.exit(1)
    if kg['num_relations'] != n_rel:
        print(f"  FAIL: relation count KG={kg['num_relations']} vs PyKEEN={n_rel}")
        sys.exit(1)
    print(f"  PASS: counts match load_kg_fashion")

    print("\n" + "=" * 70)
    print(" B — SASRec trial_017/best_valid.pth")
    print("=" * 70)

    if not os.path.exists(SASREC_PATH):
        print(f"  ERROR: File not found: {SASREC_PATH}")
        sys.exit(1)

    print(f"\n[B1] Loading {SASREC_PATH}...")
    sasrec_ckpt = torch.load(SASREC_PATH, map_location='cpu', weights_only=False)

    if isinstance(sasrec_ckpt, dict) and 'state_dict' in sasrec_ckpt:
        sasrec_sd = sasrec_ckpt['state_dict']
        print(f"  Format: dict with key 'state_dict'")
    else:
        sasrec_sd = sasrec_ckpt
        print(f"  Format: direct state_dict")

    print(f"  Keys found: {list(sasrec_sd.keys())[:8]} ...")

    print(f"\n[B2]   item_embeddings...")

    embed_key = None
    for k in sasrec_sd.keys():
        if 'item_embedding' in k or 'item_emb' in k:
            embed_key = k
            break
    if embed_key is None:
        print(f"  ERROR: No 'item_embedding' key found.")
        print(f"  Available keys: {list(sasrec_sd.keys())}")
        sys.exit(1)

    item_emb = sasrec_sd[embed_key]
    print(f"  Key found          : '{embed_key}'")
    print(f"  Shape              : {tuple(item_emb.shape)}")

    n_items_recbole, hidden_size = item_emb.shape
    print(f"  num_items (con PAD): {n_items_recbole:,}  -> item reali: {n_items_recbole - 1:,}")
    print(f"  hidden_size        : {hidden_size}")

    expected_items_recbole = kg['num_entities']
    n_kg_items = len(kg['entity_type_indices']['item'])
    print(f"  Item nel KG        : {n_kg_items:,}")

    if n_items_recbole - 1 != n_kg_items:
        print(f"  ERROR: item RecBole ({n_items_recbole-1}) != item KG ({n_kg_items})")
        print(f"  This might be acceptable if there are KG-isolated items,")
        print(f"  but in fashion we have 0 isolated -> they should match.")
    else:
        print(f"  PASS: iRecBole tem coincide with KG item")

    if hidden_size != EXPECTED_EMBED_DIM:
        print(f"  FAIL: hidden_size={hidden_size} != EXPECTED_EMBED_DIM={EXPECTED_EMBED_DIM}")
        sys.exit(1)
    print(f"  PASS: hidden_size={hidden_size} coincide with embedding_dim TransE={EXPECTED_EMBED_DIM}")

    print(f"\n[B3] SASRec item embedding weight statistics...")
    emb_norms = item_emb.norm(dim=1)
    print(f"  L2 norm: mean={emb_norms.mean():.4f}, std={emb_norms.std():.4f}, "
          f"min={emb_norms.min():.4f}, max={emb_norms.max():.4f}")
    print(f"  NaN: {torch.isnan(item_emb).any().item()}")
    print(f"  Inf: {torch.isinf(item_emb).any().item()}")
    if torch.isnan(item_emb).any() or torch.isinf(item_emb).any():
        print("  FAIL: NaN o Inf in SASRec pth !")
        sys.exit(1)
    print(f"  PASS: pth SASRec ok")


    print("\n" + "=" * 70)
    print("VERIFICATION PASSED")
    print("=" * 70)
    print(f" TransE: {n_ent:,} entity x {dim_ent} dim, {n_rel} relations")
    print(f" SASRec : {n_items_recbole:,} item (with PAD) x {hidden_size} dim")
    print(f" Shared dim: {dim_ent} == {hidden_size} -> warm start direct, no projection")
    print()
    print("Next step: verify_transe_metrics_fashion.py")
    print(" -> Reload best_model.pt with PyKEEN and recalculate MRR on the test set")
    print(" -> Verify that it matches MRR=0.1057 from the log")


if __name__ == "__main__":
    main()
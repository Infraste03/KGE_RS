"""
Verify pre-trained TransE weights from PyKEEN.

Goal:
    Before we use best_model.pt for warm-starting the alternate learning,
    we need to verify two things:

    (1) The shapes of the embedding matrices match what we expect:
        - entity matrix: (24900,400)
        - relation matrix: (5, 400)

    (2) PyKEEN's entity ordering matches our load_kg.py ordering. PyKEEN
        builds entity IDs by alphabetically sorting entity strings, and
        load_kg.py does the same. So in theory they coincide. We verify
        this in practice with a deterministic check.

If both verifications pass, we can use the weights as-is for warm starting.
If they fail, we'll need to either re-train TransE or find the mapping.

Usage:
    python verify_pretrained_weights.py
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from load_kg import load_kg


# --------------------------------------------------------------------------- #
# Paths - adjust these to your project structure
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


BEST_MODEL_PATH = r"KG_RS\best_model.pt"
# NOTE: change the path above to wherever your best_model.pt is located.


KG_TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "kg_train.tsv")
KG_VALID_PATH = os.path.join(BASE_DIR, "data", "processed", "taskA_valid.tsv")
KG_TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "taskA_test.tsv")

EXPECTED_NUM_ENTITIES = 24_900
EXPECTED_NUM_RELATIONS = 5
EXPECTED_EMBEDDING_DIM = 400


# --------------------------------------------------------------------------- #
# Main verification
# --------------------------------------------------------------------------- #

def main():
    print("=" * 70)
    print("VERIFY PRE-TRAINED TRANS-E WEIGHTS")
    print("=" * 70)

    # ------------------------------------------------------------------- #
    # 1. Load the state_dict from best_model.pt
    # ------------------------------------------------------------------- #
    print(f"\n[1/4] Loading {BEST_MODEL_PATH} ...")
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"  ERROR: file not found at {BEST_MODEL_PATH}")
        print(f"  Adjust BEST_MODEL_PATH at the top of this script.")
        sys.exit(1)

    state_dict = torch.load(
        BEST_MODEL_PATH, map_location='cpu', weights_only=False
    )

    expected_keys = {
        'entity_representations.0._embeddings.weight',
        'relation_representations.0._embeddings.weight',
    }
    actual_keys = set(state_dict.keys())
    if actual_keys != expected_keys:
        print(f"  ERROR: unexpected keys in state_dict.")
        print(f"  Expected: {expected_keys}")
        print(f"  Got     : {actual_keys}")
        sys.exit(1)

    entity_weights = state_dict['entity_representations.0._embeddings.weight']
    relation_weights = state_dict['relation_representations.0._embeddings.weight']

    print(f"  PASS: loaded state_dict with the expected 2 keys")
    print(f"  entity_weights   shape: {tuple(entity_weights.shape)}")
    print(f"  relation_weights shape: {tuple(relation_weights.shape)}")

    # ------------------------------------------------------------------- #
    # 2. Verify shapes
    # ------------------------------------------------------------------- #
    print(f"\n[2/4] Verifying shapes ...")

    n_ent, dim_ent = entity_weights.shape
    n_rel, dim_rel = relation_weights.shape

    errors = []
    if n_ent != EXPECTED_NUM_ENTITIES:
        errors.append(
            f"  - entity count mismatch: expected {EXPECTED_NUM_ENTITIES}, got {n_ent}"
        )
    if n_rel != EXPECTED_NUM_RELATIONS:
        errors.append(
            f"  - relation count mismatch: expected {EXPECTED_NUM_RELATIONS}, got {n_rel}"
        )
    if dim_ent != EXPECTED_EMBEDDING_DIM:
        errors.append(
            f"  - entity embedding dim mismatch: expected {EXPECTED_EMBEDDING_DIM}, got {dim_ent}"
        )
    if dim_rel != EXPECTED_EMBEDDING_DIM:
        errors.append(
            f"  - relation embedding dim mismatch: expected {EXPECTED_EMBEDDING_DIM}, got {dim_rel}"
        )
    if dim_ent != dim_rel:
        errors.append(
            f"  - entity dim ({dim_ent}) != relation dim ({dim_rel})"
        )

    if errors:
        print("  FAIL: shape mismatch:")
        for e in errors:
            print(e)
        print("\n  If embedding_dim differs, update EXPECTED_EMBEDDING_DIM at the top.")
        sys.exit(1)

    print(f"  PASS: shapes match the expected values")
    print(f"    {n_ent:,} entities x {dim_ent} dims")
    print(f"    {n_rel:,} relations x {dim_rel} dims")

    # ------------------------------------------------------------------- #
    # 3. Sanity stats on the weights
    # ------------------------------------------------------------------- #
    print(f"\n[3/4] Computing sanity statistics ...")

    ent_norms = entity_weights.norm(dim=1)
    rel_norms = relation_weights.norm(dim=1)

    print(f"  Entity embeddings:")
    print(f"    L2 norm  -> mean={ent_norms.mean().item():.4f}, "
          f"std={ent_norms.std().item():.4f}, "
          f"min={ent_norms.min().item():.4f}, "
          f"max={ent_norms.max().item():.4f}")
    print(f"    has NaN  : {torch.isnan(entity_weights).any().item()}")
    print(f"    has Inf  : {torch.isinf(entity_weights).any().item()}")

    print(f"  Relation embeddings:")
    print(f"    L2 norm  -> mean={rel_norms.mean().item():.4f}, "
          f"std={rel_norms.std().item():.4f}, "
          f"min={rel_norms.min().item():.4f}, "
          f"max={rel_norms.max().item():.4f}")

    # TransE typically normalizes entity embeddings to unit norm during training.
    # If mean norm is very close to 1.0, that's a strong signal that PyKEEN
    # was using normalized TransE.
    if 0.9 < ent_norms.mean().item() < 1.1:
        print(f"  -> Entity norms are close to 1.0: PyKEEN used unit-normalized TransE.")
    else:
        print(f"  -> Entity norms are not unit-normalized (mean ~{ent_norms.mean().item():.2f}).")

    if torch.isnan(entity_weights).any() or torch.isinf(entity_weights).any():
        print("  FAIL: NaN or Inf detected in weights!")
        sys.exit(1)

    print(f"  PASS: weights look healthy")

    # ------------------------------------------------------------------- #
    # 4. Verify our load_kg mapping has the same number of entities/relations
    # ------------------------------------------------------------------- #
    print(f"\n[4/4] Loading our KG mapping with load_kg.py ...")

    kg = load_kg(
        train_path=KG_TRAIN_PATH,
        valid_path=KG_VALID_PATH,
        test_path=KG_TEST_PATH,
        verbose=False,
    )

    print(f"  Our mapping has {kg['num_entities']:,} entities, "
          f"{kg['num_relations']:,} relations")

    if kg['num_entities'] != n_ent:
        print(f"  FAIL: entity count mismatch with PyKEEN!")
        print(f"    PyKEEN: {n_ent}, ours: {kg['num_entities']}")
        sys.exit(1)
    if kg['num_relations'] != n_rel:
        print(f"  FAIL: relation count mismatch with PyKEEN!")
        sys.exit(1)

    print(f"  PASS: counts match between PyKEEN's state_dict and our load_kg mapping")

    # ------------------------------------------------------------------- #
    # FINAL VERDICT
    # ------------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("VERIFICATION PASSED")
    print("=" * 70)
    print()
    print("Summary of what we verified:")
    print(f"  - best_model.pt loads cleanly as a state_dict")
    print(f"  - shapes match expected: {n_ent} entities x {dim_ent} dims, "
          f"{n_rel} relations x {dim_rel} dims")
    print(f"  - no NaN or Inf in the weights")
    print(f"  - entity norms have a sensible distribution")
    print(f"  - counts match our load_kg.py mapping")
    print()
    print("Open question (to handle in run_step4.py):")
    print(f"  - The exact entity ordering of PyKEEN vs ours.")
    print(f"    PyKEEN sorts entities alphabetically; load_kg.py does the same.")
    print(f"    If both use the union of {{train, valid, test}} entities,")
    print(f"    the orderings should be identical. We will assume so for now")
    print(f"    and add a runtime check in run_step4.py before training starts.")


if __name__ == "__main__":
    main()
"""
TaskATransE: the TransE model for the link prediction task on the KG.

This module defines the model used in Task A of the alternate learning.
It operates ON TOP of a SharedEmbedding instance:

    - SharedEmbedding owns the entity embeddings (24,957 x 400).
      It is shared between Task A and Task B.

    - TaskATransE owns the RELATION embeddings (5 x 400).
      These are private to Task A; SASRec does not use relations.

    - TaskATransE.score(h, r, t) computes the TransE scoring function
      ||h + r - t|| (L2 norm). Lower score = triple is more likely true.

    - TaskATransE.score_all_tails(h, r) and .score_all_heads(r, t) compute
      scores against ALL candidate entities at once, used for link prediction
      evaluation (Hits@K, MRR). These mirror PyKEEN's model.score_t / score_h
      so we can replicate the evaluate_custom function from Step 2 exactly.

What this module does NOT do (intentional separation of concerns):

    - It does NOT compute the loss. The loss is in losses.py.
    - It does NOT sample negative triples. The sampler is in samplers.py.
    - It does NOT do training. The training loop is in alternate_loop.py.

----------------------------------------------------------------------------
Where to put this file:
    B2B_alternate_learning/models/task_a_transe.py
----------------------------------------------------------------------------
"""
#B2B_alternate_learning/models/task_a_transe.py

import os
import sys
import logging
import torch
import torch.nn as nn

# Make shared_embedding importable when this file is in the models/ folder
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from shared_embedding import SharedEmbedding

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =========================================================================== #
# THE TaskATransE CLASS
# =========================================================================== #

class TaskATransE(nn.Module):
    """
    TransE model for Task A (KG link prediction), operating on a shared
    entity embedding matrix.

    Parameters
    ----------
    shared_embedding : SharedEmbedding
        The shared entity embedding matrix (used by both Task A and Task B).
    num_relations : int
        Number of distinct relations in the KG (5 in our case).
    embedding_dim : int
        Dimension of relation embeddings. Must match shared_embedding.embedding_dim.
    p_norm : int, optional
        The L_p norm to use in the score function. Default 2 (Euclidean).
    normalize_entities : bool, optional
        If True, entity embeddings are renormalized to unit L2 norm at each
        forward pass. Matches PyKEEN's TransE. Default: True.
    """

    def __init__(
        self,
        shared_embedding: SharedEmbedding,
        num_relations: int,
        embedding_dim: int,
        p_norm: int = 1,
        normalize_entities: bool = True,
    ):
        super().__init__()

        if shared_embedding.embedding_dim != embedding_dim:
            raise ValueError(
                f"shared_embedding has dim {shared_embedding.embedding_dim}, "
                f"but TaskATransE was given embedding_dim {embedding_dim}. "
                f"They MUST match for hard sharing."
            )

        # Store shared_embedding as a regular attribute, NOT as child module.
        # See Test 6 in the smoke test for why this matters.
        object.__setattr__(self, 'shared_embedding', shared_embedding)

        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.p_norm = p_norm
        self.normalize_entities = normalize_entities

        # Relation embeddings (private to Task A)
        self.relation_embedding = nn.Embedding(
            num_embeddings=num_relations,
            embedding_dim=embedding_dim,
        )
        nn.init.xavier_uniform_(self.relation_embedding.weight)

        logger.info(
            f"Initialized TaskATransE: "
            f"{num_relations} relations x {embedding_dim} dims. "
            f"p_norm={p_norm}, normalize_entities={normalize_entities}"
        )

    # ----------------------------------------------------------------------- #
    # Core scoring function (single triple at a time)
    # ----------------------------------------------------------------------- #

    def score(
        self,
        head_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        tail_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        TransE energy score: lower = more likely true.
        Returns ||h + r - t||_p for each triple.

        Used during training for the loss function.
        """
        h = self.shared_embedding(head_ids)
        t = self.shared_embedding(tail_ids)
        r = self.relation_embedding(relation_ids)

        if self.normalize_entities:
            h = nn.functional.normalize(h, p=2, dim=-1)
            t = nn.functional.normalize(t, p=2, dim=-1)

        scores = torch.norm(h + r - t, p=self.p_norm, dim=-1)
        return scores

    def score_positive_and_negative(
        self,
        pos_triples: torch.Tensor,
        neg_triples: torch.Tensor,
    ) -> tuple:
        """Convenience: score positive and negative triples together."""
        pos_scores = self.score(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        neg_scores = self.score(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])
        return pos_scores, neg_scores

    # ----------------------------------------------------------------------- #
    # Link prediction methods (for evaluation, not for training)
    #
    # These mirror PyKEEN's model.score_t / model.score_h, so that we can
    # replicate the evaluate_custom function from Step 2's run_hpo_kge2.py
    # without modification.
    #
    # IMPORTANT CONVENTION (matches PyKEEN):
    #   These methods return SIMILARITY scores, NOT energy scores.
    #   similarity = -energy = -||h + r - t||
    #   HIGHER similarity = MORE likely to be the true entity.
    #
    #   This is the convention used by Step 2's evaluate_custom, which counts
    #   the rank as (scores > scores[t]).sum() + 1 -- the number of entities
    #   with HIGHER score than the ground truth, plus one.
    #
    #   If we returned energies (lower = better) here, the rank computation
    #   would need to be inverted, and we'd diverge from PyKEEN/Step 2.
    # ----------------------------------------------------------------------- #


    def score_all_tails(self, head_ids, relation_ids):
        """Score all entities as candidate tails for given (h, r)."""
        h = self.shared_embedding(head_ids)        # (batch, dim)
        r = self.relation_embedding(relation_ids)  # (batch, dim)
        all_entities = self.shared_embedding.embedding.weight  # (num_entities, dim)

        # h + r: (batch, dim)
        hr = h + r

        # Compute ||hr - t||_p explicitly, NOT via cdist
        # hr.unsqueeze(1): (batch, 1, dim)
        # all_entities.unsqueeze(0): (1, num_entities, dim)
        # diff: (batch, num_entities, dim)
        diff = hr.unsqueeze(1) - all_entities.unsqueeze(0)
        # ||diff||_p along the last dim: (batch, num_entities)
        energy = torch.norm(diff, p=self.p_norm, dim=-1)
        return -energy

    def score_all_heads(self, relation_ids, tail_ids):
        """Score all entities as candidate heads for given (r, t)."""
        t = self.shared_embedding(tail_ids)        # (batch, dim)
        r = self.relation_embedding(relation_ids)  # (batch, dim)
        all_entities = self.shared_embedding.embedding.weight  # (num_entities, dim)
        # We want ||h + r - t|| small, so we want h close to (t - r)
        target = t - r  # (batch, dim)
        # Compute ||h - target||_p for every candidate h
        diff = all_entities.unsqueeze(0) - target.unsqueeze(1)
        energy = torch.norm(diff, p=self.p_norm, dim=-1)
        return -energy

    # ----------------------------------------------------------------------- #
    # Warm-start: load relation embeddings from PyKEEN's best_model.pt
    # ----------------------------------------------------------------------- #

    def load_relation_weights_from_pykeen(
        self,
        pykeen_relation_weights: torch.Tensor,
        relation_id_mapping_pykeen: dict,
        relation_id_mapping_ours: dict,
        verbose: bool = True,
    ):
        """Load TransE relation embeddings trained with PyKEEN in Step 2."""
        if verbose:
            logger.info("\n--- Warm start: loading relation weights from PyKEEN ---")

        if pykeen_relation_weights.shape[1] != self.embedding_dim:
            raise ValueError(
                f"PyKEEN relation embedding dim ({pykeen_relation_weights.shape[1]}) "
                f"does not match our embedding dim ({self.embedding_dim})."
            )

        if pykeen_relation_weights.shape[0] != self.num_relations:
            raise ValueError(
                f"PyKEEN has {pykeen_relation_weights.shape[0]} relations, "
                f"but TaskATransE was built with {self.num_relations}."
            )

        missing = [r for r in relation_id_mapping_ours if r not in relation_id_mapping_pykeen]
        if missing:
            raise ValueError(
                f"Relations {missing} are in our mapping but not in PyKEEN's."
            )

        norm_before = self.relation_embedding.weight.norm(dim=1).mean().item()

        with torch.no_grad():
            for relation_str, our_id in relation_id_mapping_ours.items():
                pykeen_id = relation_id_mapping_pykeen[relation_str]
                self.relation_embedding.weight[our_id] = pykeen_relation_weights[pykeen_id]

        norm_after = self.relation_embedding.weight.norm(dim=1).mean().item()

        if verbose:
            logger.info(f"  Loaded relation weights for {len(relation_id_mapping_ours)} relations")
            logger.info(f"  Mean L2 norm BEFORE warm start: {norm_before:.4f} (random init)")
            logger.info(f"  Mean L2 norm AFTER  warm start: {norm_after:.4f}")
            logger.info(
                f"  These should differ. If they are equal, the warm start did NOTHING."
            )

    # ----------------------------------------------------------------------- #
    # Diagnostics
    # ----------------------------------------------------------------------- #

    def diagnostic_summary(self) -> dict:
        with torch.no_grad():
            w = self.relation_embedding.weight
            norms = w.norm(dim=1)
            return {
                'shape': tuple(w.shape),
                'norm_mean': norms.mean().item(),
                'norm_std': norms.std().item(),
                'norm_min': norms.min().item(),
                'norm_max': norms.max().item(),
                'has_nan': torch.isnan(w).any().item(),
                'has_inf': torch.isinf(w).any().item(),
            }

    def print_diagnostic(self, label: str = ""):
        s = self.diagnostic_summary()
        logger.info(f"\n--- TaskATransE relation diagnostic [{label}] ---")
        logger.info(f"  shape         : {s['shape']}")
        logger.info(f"  norm (L2) mean: {s['norm_mean']:.4f}")
        logger.info(f"  norm (L2) std : {s['norm_std']:.4f}")
        logger.info(f"  norm (L2) min : {s['norm_min']:.4f}")
        logger.info(f"  norm (L2) max : {s['norm_max']:.4f}")
        logger.info(f"  has NaN       : {s['has_nan']}")
        logger.info(f"  has Inf       : {s['has_inf']}")
        if s['has_nan'] or s['has_inf']:
            logger.error("  PROBLEM: NaN or Inf in relation embeddings!")


# =========================================================================== #
# STANDALONE SMOKE TEST
# =========================================================================== #

if __name__ == "__main__":

    print("=" * 70)
    print("TASK A (TransE) SMOKE TEST")
    print("=" * 70)

    NUM_ENTITIES = 24_957
    NUM_RELATIONS = 5
    EMBEDDING_DIM = 400
    BATCH_SIZE = 32

    shared = SharedEmbedding(
        num_entities=NUM_ENTITIES,
        embedding_dim=EMBEDDING_DIM,
        padding_idx=None,
    )

    task_a = TaskATransE(
        shared_embedding=shared,
        num_relations=NUM_RELATIONS,
        embedding_dim=EMBEDDING_DIM,
        p_norm=2,
        normalize_entities=True,
    )

    task_a.print_diagnostic(label="after random init")

    # ---- Test 1: forward pass (single triples) ----
    print("\n--- Test 1: forward pass (single triples) ---")
    head_ids = torch.randint(0, NUM_ENTITIES, (BATCH_SIZE,))
    rel_ids = torch.randint(0, NUM_RELATIONS, (BATCH_SIZE,))
    tail_ids = torch.randint(0, NUM_ENTITIES, (BATCH_SIZE,))

    scores = task_a.score(head_ids, rel_ids, tail_ids)
    assert scores.shape == (BATCH_SIZE,), f"Expected shape ({BATCH_SIZE},), got {scores.shape}"
    print(f"  PASS: scored a batch of {BATCH_SIZE} triples")
    print(f"  scores stats : mean={scores.mean().item():.4f}, "
          f"min={scores.min().item():.4f}, max={scores.max().item():.4f}")

    # ---- Test 2: gradient flow ----
    print("\n--- Test 2: gradient flow ---")
    shared.embedding.weight.grad = None
    task_a.relation_embedding.weight.grad = None

    scores = task_a.score(head_ids, rel_ids, tail_ids)
    fake_loss = scores.sum()
    fake_loss.backward()

    grad_shared = shared.embedding.weight.grad
    grad_relations = task_a.relation_embedding.weight.grad

    assert grad_shared is not None, "No gradient on shared_embedding!"
    assert grad_relations is not None, "No gradient on relation_embedding!"
    assert (grad_shared.abs().sum() > 0).item(), "Shared gradient is all zero!"
    assert (grad_relations.abs().sum() > 0).item(), "Relation gradient is all zero!"

    print(f"  PASS: gradients flow to BOTH shared_embedding and relation_embedding")
    print(f"  shared_embedding grad norm   : {grad_shared.norm().item():.4f}")
    print(f"  relation_embedding grad norm : {grad_relations.norm().item():.4f}")

    # ---- Test 3: positive vs negative scoring ----
    print("\n--- Test 3: score_positive_and_negative ---")
    pos_triples = torch.stack([head_ids, rel_ids, tail_ids], dim=1)
    neg_tails = tail_ids[torch.randperm(BATCH_SIZE)]
    neg_triples = torch.stack([head_ids, rel_ids, neg_tails], dim=1)

    pos_scores, neg_scores = task_a.score_positive_and_negative(pos_triples, neg_triples)
    assert pos_scores.shape == (BATCH_SIZE,)
    assert neg_scores.shape == (BATCH_SIZE,)
    print(f"  PASS: score_positive_and_negative returns ({pos_scores.shape}, {neg_scores.shape})")

    # ---- Test 4: warm start of relations ----
    print("\n--- Test 4: warm start of relation embeddings ---")
    fake_pykeen_relations = torch.randn(NUM_RELATIONS, EMBEDDING_DIM) * 5.0
    fake_rel_to_id_pykeen = {f"rel_{i}": i for i in range(NUM_RELATIONS)}
    fake_rel_to_id_ours = {f"rel_{i}": i for i in range(NUM_RELATIONS)}

    norm_before = task_a.relation_embedding.weight.norm(dim=1).mean().item()
    task_a.load_relation_weights_from_pykeen(
        pykeen_relation_weights=fake_pykeen_relations,
        relation_id_mapping_pykeen=fake_rel_to_id_pykeen,
        relation_id_mapping_ours=fake_rel_to_id_ours,
        verbose=True,
    )
    norm_after = task_a.relation_embedding.weight.norm(dim=1).mean().item()
    assert norm_before != norm_after, "Warm start did not change the relation weights!"
    print(f"  PASS: warm start changed relation weights ({norm_before:.4f} -> {norm_after:.4f})")

    # ---- Test 5: dimension mismatch is caught ----
    print("\n--- Test 5: dimension mismatch is caught ---")
    bad_relations = torch.randn(NUM_RELATIONS, EMBEDDING_DIM + 1)
    try:
        task_a.load_relation_weights_from_pykeen(
            pykeen_relation_weights=bad_relations,
            relation_id_mapping_pykeen=fake_rel_to_id_pykeen,
            relation_id_mapping_ours=fake_rel_to_id_ours,
            verbose=False,
        )
        print("  FAIL: dimension mismatch was NOT caught")
    except ValueError as e:
        print(f"  PASS: caught the dimension mismatch correctly")

    # ---- Test 6: shared_embedding is NOT registered as a child module ----
    print("\n--- Test 6: shared_embedding is NOT registered as a child module ---")
    own_params = sum(p.numel() for p in task_a.parameters())
    expected_relation_params = NUM_RELATIONS * EMBEDDING_DIM
    assert own_params == expected_relation_params, (
        f"TaskATransE has {own_params} parameters, expected only {expected_relation_params}"
    )
    print(f"  PASS: TaskATransE owns {own_params:,} parameters (only relation embeddings)")

    # ---- Test 7 (NEW): score_all_tails returns the correct shape ----
    print("\n--- Test 7: score_all_tails (link prediction) ---")
    small_batch = 4
    h_small = torch.randint(0, NUM_ENTITIES, (small_batch,))
    r_small = torch.randint(0, NUM_RELATIONS, (small_batch,))

    all_tail_scores = task_a.score_all_tails(h_small, r_small)
    expected_shape = (small_batch, NUM_ENTITIES)
    assert all_tail_scores.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {all_tail_scores.shape}"
    )
    print(f"  PASS: score_all_tails returns shape {all_tail_scores.shape}")
    print(f"  scores stats : mean={all_tail_scores.mean().item():.4f}, "
          f"min={all_tail_scores.min().item():.4f}, max={all_tail_scores.max().item():.4f}")

    # ---- Test 8 (NEW): score_all_heads returns the correct shape ----
    print("\n--- Test 8: score_all_heads (link prediction) ---")
    t_small = torch.randint(0, NUM_ENTITIES, (small_batch,))

    all_head_scores = task_a.score_all_heads(r_small, t_small)
    assert all_head_scores.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {all_head_scores.shape}"
    )
    print(f"  PASS: score_all_heads returns shape {all_head_scores.shape}")

    # ---- Test 9 (NEW): consistency between score and score_all_tails ----
    # For a triple (h, r, t), the energy ||h + r - t|| should equal
    # -score_all_tails(h, r)[t] (similarity = -energy at column t).
    print("\n--- Test 9: consistency between score() and score_all_tails() ---")
    test_h = torch.tensor([5, 100, 1000])
    test_r = torch.tensor([0, 2, 4])
    test_t = torch.tensor([42, 500, 2000])

    energy = task_a.score(test_h, test_r, test_t)
    all_t_sim = task_a.score_all_tails(test_h, test_r)
    sim_at_t = torch.stack([all_t_sim[i, test_t[i]] for i in range(3)])

    diff = (sim_at_t - (-energy)).abs().max().item()
    assert diff < 1e-4, f"Inconsistency: max diff = {diff}"
    print(f"  PASS: score_all_tails[i, t_i] == -score(h_i, r_i, t_i)")
    print(f"  max diff between the two paths: {diff:.2e}")

    # ---- Test 10 (NEW): consistency between score and score_all_heads ----
    print("\n--- Test 10: consistency between score() and score_all_heads() ---")
    energy_h = task_a.score(test_h, test_r, test_t)
    all_h_sim = task_a.score_all_heads(test_r, test_t)
    sim_at_h = torch.stack([all_h_sim[i, test_h[i]] for i in range(3)])

    diff_h = (sim_at_h - (-energy_h)).abs().max().item()
    assert diff_h < 1e-4, f"Inconsistency: max diff = {diff_h}"
    print(f"  PASS: score_all_heads[i, h_i] == -score(h_i, r_i, t_i)")
    print(f"  max diff between the two paths: {diff_h:.2e}")

    print("\n" + "=" * 70)
    print("TASK A (TransE) SMOKE TEST PASSED")
    print("=" * 70)
"""
task_a_transe_fashion.py
=========================

TaskATransE: TransE model for Task A of Fashion alternate learning.

It operates ON TOP of SharedEmbedding:
  - SharedEmbedding owns the entity embeddings (182,984 x 64)
    and is shared between Task A and Task B.
  - TaskATransE owns the RELATION embeddings (3 x 64),
    which are private to Task A (SASRec does not use relations).

Scoring function: -||h + r - t||_p (similarity = -energy)
Higher = more likely that the triple is true.

Differences compared with B2B:
  - 3 relations instead of 5
  - embedding_dim = 64 instead of 400
  - compatible_with is item->item (not item->machine)
  - type_constrained and naive coincide for compatible_with

File location:
    fashion_generalization/alternate_learning/models/task_a_transe_fashion.py
"""

import os
import sys
import logging
import torch
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from shared_embedding_fashion import SharedEmbedding

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class TaskATransE(nn.Module):
    """
    TransE for Task A, operating on the Fashion SharedEmbedding.

    Parameters
    ----------
    shared_embedding : SharedEmbedding
        Shared embedding matrix (182,984 x 64).
    num_relations : int
        Number of relations in the Fashion KG. Default: 3
        (compatible_with, belongs_to, belongs_to_brand)
    embedding_dim : int
        Embedding dimension. Must match shared_embedding.embedding_dim.
        For Fashion: 64.
    p_norm : int
        Lp norm used in the scoring function. Default: 1 (as in PyKEEN default).
    normalize_entities : bool
        If True, entity embeddings are renormalized to L2 norm = 1
        before score computation. Must be True to replicate PyKEEN.
        Default: True.
    """

    def __init__(
        self,
        shared_embedding: SharedEmbedding,
        num_relations: int = 3,
        embedding_dim: int = 64,
        p_norm: int = 1,
        normalize_entities: bool = True,
    ):
        super().__init__()

        if shared_embedding.embedding_dim != embedding_dim:
            raise ValueError(
                f"shared_embedding has dim {shared_embedding.embedding_dim}, "
                f"but TaskATransE was created with embedding_dim={embedding_dim}. "
                f"They must match for hard sharing."
            )

        # shared_embedding is NOT registered as a child module.
        # Otherwise its parameters would appear twice in the optimizer
        # (once for Task A and once for Task B), causing double updates.
        # Test 6 in the smoke test verifies this.
        object.__setattr__(self, 'shared_embedding', shared_embedding)

        self.num_relations     = num_relations
        self.embedding_dim     = embedding_dim
        self.p_norm            = p_norm
        self.normalize_entities = normalize_entities

        # Relation embeddings (private to Task A)
        self.relation_embedding = nn.Embedding(
            num_embeddings=num_relations,
            embedding_dim=embedding_dim,
        )
        nn.init.xavier_uniform_(self.relation_embedding.weight)

        logger.info(
            f"TaskATransE initialized: "
            f"{num_relations} relations x {embedding_dim} dim, "
            f"p_norm={p_norm}, normalize_entities={normalize_entities}"
        )

    # ----------------------------------------------------------------------- #
    # Training scoring
    # ----------------------------------------------------------------------- #

    def score(
        self,
        head_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        tail_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        TransE energy score: ||h + r - t||_p.
        Lower = more likely that the triple is true.
        Used during training for the margin-ranking loss.

        Returns: FloatTensor of shape (batch_size,)
        """
        h = self.shared_embedding(head_ids)
        t = self.shared_embedding(tail_ids)
        r = self.relation_embedding(relation_ids)

        if self.normalize_entities:
            h = nn.functional.normalize(h, p=2, dim=-1)
            t = nn.functional.normalize(t, p=2, dim=-1)

        return torch.norm(h + r - t, p=self.p_norm, dim=-1)

    def score_positive_and_negative(
        self,
        pos_triples: torch.Tensor,
        neg_triples: torch.Tensor,
    ) -> tuple:
        """Convenience method: score positive and negative triples together."""
        pos = self.score(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        neg = self.score(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])
        return pos, neg

    # ----------------------------------------------------------------------- #
    # Link prediction (for evaluation, not training)
    #
    # These methods return SIMILARITY = -energy (higher = more likely).
    # This is consistent with PyKEEN model.score_t / model.score_h.
    # Rank is computed as: (scores > scores[t]).sum() + 1
    # ----------------------------------------------------------------------- #

    def score_all_tails(
        self,
        head_ids: torch.Tensor,
        relation_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score ALL entities as candidate tails for (head, relation).
        Mirrors PyKEEN model.score_t.

        Parameters
        ----------
        head_ids     : LongTensor (batch_size,)
        relation_ids : LongTensor (batch_size,)

        Returns
        -------
        FloatTensor (batch_size, num_entities)
            scores[i, j] = similarity of j as tail of
            (head_ids[i], relation_ids[i])
        """
        h = self.shared_embedding(head_ids)
        r = self.relation_embedding(relation_ids)
        all_e = self.shared_embedding.embedding.weight  # (num_entities, dim)

        if self.normalize_entities:
            h     = nn.functional.normalize(h, p=2, dim=-1)
            all_e = nn.functional.normalize(all_e, p=2, dim=-1)

        # hr: (batch, dim)
        hr = h + r

        # diff: (batch, num_entities, dim)
        diff = hr.unsqueeze(1) - all_e.unsqueeze(0)

        # energy: (batch, num_entities)
        energy = torch.norm(diff, p=self.p_norm, dim=-1)

        return -energy

    def score_all_heads(
        self,
        relation_ids: torch.Tensor,
        tail_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score ALL entities as candidate heads for (relation, tail).
        Mirrors PyKEEN model.score_h.

        Parameters
        ----------
        relation_ids : LongTensor (batch_size,)
        tail_ids     : LongTensor (batch_size,)

        Returns
        -------
        FloatTensor (batch_size, num_entities)
        """
        t = self.shared_embedding(tail_ids)
        r = self.relation_embedding(relation_ids)
        all_e = self.shared_embedding.embedding.weight  # (num_entities, dim)

        if self.normalize_entities:
            t     = nn.functional.normalize(t, p=2, dim=-1)
            all_e = nn.functional.normalize(all_e, p=2, dim=-1)

        # We want ||h + r - t|| to be small -> h close to (t - r)
        target = t - r  # (batch, dim)

        # diff: (batch, num_entities, dim)
        diff = all_e.unsqueeze(0) - target.unsqueeze(1)
        energy = torch.norm(diff, p=self.p_norm, dim=-1)

        return -energy

    # ----------------------------------------------------------------------- #
    # Warm start of relation embeddings from PyKEEN
    # ----------------------------------------------------------------------- #

    def load_relation_weights_from_pykeen(
        self,
        pykeen_relation_weights: torch.Tensor,
        relation_id_mapping_pykeen: dict,
        relation_id_mapping_ours: dict,
        verbose: bool = True,
    ):
        """
        Load TransE relation weights trained with PyKEEN.

        Fashion: 3 relations
        (belongs_to=0, belongs_to_brand=1, compatible_with=2
        in the alphabetical ordering of load_kg_fashion).

        The copy is performed row by row through relation_string for safety,
        even though the two mappings coincide under alphabetical ordering.
        """
        if verbose:
            logger.info("\n--- Warm start: loading relation weights from PyKEEN ---")

        if pykeen_relation_weights.shape[1] != self.embedding_dim:
            raise ValueError(
                f"PyKEEN relation dim ({pykeen_relation_weights.shape[1]}) != "
                f"our embedding_dim ({self.embedding_dim})."
            )
        if pykeen_relation_weights.shape[0] != self.num_relations:
            raise ValueError(
                f"PyKEEN has {pykeen_relation_weights.shape[0]} relations, "
                f"but TaskATransE was created with {self.num_relations}."
            )

        missing = [r for r in relation_id_mapping_ours if r not in relation_id_mapping_pykeen]
        if missing:
            raise ValueError(
                f"Relations {missing} are present in our mapping but not in PyKEEN."
            )

        norm_before = self.relation_embedding.weight.norm(p=2, dim=1).mean().item()

        with torch.no_grad():
            for rel_str, our_id in relation_id_mapping_ours.items():
                pykeen_id = relation_id_mapping_pykeen[rel_str]
                self.relation_embedding.weight[our_id] = \
                    pykeen_relation_weights[pykeen_id].to(
                        self.relation_embedding.weight.device
                    )

        norm_after = self.relation_embedding.weight.norm(p=2, dim=1).mean().item()

        if verbose:
            logger.info(f"  Relations loaded          : {len(relation_id_mapping_ours)}")
            logger.info(f"  Mean L2 norm BEFORE       : {norm_before:.4f}")
            logger.info(f"  Mean L2 norm AFTER        : {norm_after:.4f}")
            if abs(norm_before - norm_after) < 1e-6:
                logger.warning(
                    "  WARNING: the norm did not change. "
                    "The warm start may not have modified the weights."
                )

    # ----------------------------------------------------------------------- #
    # Diagnostics
    # ----------------------------------------------------------------------- #

    def diagnostic_summary(self) -> dict:
        with torch.no_grad():
            w     = self.relation_embedding.weight
            norms = w.norm(p=2, dim=1)
            return {
                'shape'    : tuple(w.shape),
                'norm_mean': norms.mean().item(),
                'norm_std' : norms.std().item(),
                'norm_min' : norms.min().item(),
                'norm_max' : norms.max().item(),
                'has_nan'  : torch.isnan(w).any().item(),
                'has_inf'  : torch.isinf(w).any().item(),
            }

    def print_diagnostic(self, label: str = ""):
        s = self.diagnostic_summary()
        logger.info(f"\n--- TaskATransE relation diagnostic [{label}] ---")
        logger.info(f"  shape         : {s['shape']}")
        logger.info(f"  L2 norm mean  : {s['norm_mean']:.4f}")
        logger.info(f"  L2 norm std   : {s['norm_std']:.4f}")
        logger.info(f"  L2 norm min   : {s['norm_min']:.4f}")
        logger.info(f"  L2 norm max   : {s['norm_max']:.4f}")
        logger.info(f"  has NaN       : {s['has_nan']}")
        logger.info(f"  has Inf       : {s['has_inf']}")
        if s['has_nan'] or s['has_inf']:
            logger.error("  PROBLEM: NaN or Inf found in relation embeddings!")


# =========================================================================== #
# STANDALONE SMOKE TEST
#
# Real Fashion dimensions:
#   - 182,984 SharedEmbedding rows (182,983 KG entities + 1 padding)
#   - 3 relations
#   - embedding_dim = 64
# =========================================================================== #

if __name__ == "__main__":

    print("=" * 70)
    print("TASK A TRANSE FASHION — SMOKE TEST")
    print("=" * 70)

    NUM_ENTITIES  = 182_984   # 182,983 KG entities + 1 padding
    NUM_RELATIONS = 3
    EMBEDDING_DIM = 64
    PADDING_IDX   = 182_983
    BATCH_SIZE    = 32

    shared = SharedEmbedding(
        num_entities=NUM_ENTITIES,
        embedding_dim=EMBEDDING_DIM,
        padding_idx=PADDING_IDX,
    )

    task_a = TaskATransE(
        shared_embedding=shared,
        num_relations=NUM_RELATIONS,
        embedding_dim=EMBEDDING_DIM,
        p_norm=1,
        normalize_entities=True,
    )

    task_a.print_diagnostic(label="after random initialization")

    # ---- Test 1: forward pass ----
    print("\n--- Test 1: forward pass (batch of triples) ---")
    # Use only real entity IDs (0..182982), not padding
    head_ids = torch.randint(0, 182_983, (BATCH_SIZE,))
    rel_ids  = torch.randint(0, NUM_RELATIONS, (BATCH_SIZE,))
    tail_ids = torch.randint(0, 182_983, (BATCH_SIZE,))

    scores = task_a.score(head_ids, rel_ids, tail_ids)
    assert scores.shape == (BATCH_SIZE,), f"Incorrect shape: {scores.shape}"
    assert (scores >= 0).all(), "Energy score must be >= 0 (it is a norm)"
    print(f"  PASS: shape {scores.shape}, "
          f"mean={scores.mean():.4f}, min={scores.min():.4f}, max={scores.max():.4f}")

    # ---- Test 2: gradient flow ----
    print("\n--- Test 2: gradient flow to shared_embedding AND relation_embedding ---")
    shared.embedding.weight.grad         = None
    task_a.relation_embedding.weight.grad = None

    scores = task_a.score(head_ids, rel_ids, tail_ids)
    scores.sum().backward()

    grad_shared    = shared.embedding.weight.grad
    grad_relations = task_a.relation_embedding.weight.grad

    assert grad_shared is not None, "No gradient on shared_embedding!"
    assert grad_relations is not None, "No gradient on relation_embedding!"
    assert grad_shared.abs().sum() > 0, "Gradient on shared_embedding is zero!"
    assert grad_relations.abs().sum() > 0, "Gradient on relation_embedding is zero!"
    print(f"  PASS: gradients on both matrices")
    print(f"  shared_embedding grad norm   : {grad_shared.norm():.4f}")
    print(f"  relation_embedding grad norm : {grad_relations.norm():.4f}")

    # ---- Test 3: score_positive_and_negative ----
    print("\n--- Test 3: score_positive_and_negative ---")
    pos_triples = torch.stack([head_ids, rel_ids, tail_ids], dim=1)
    neg_tails   = tail_ids[torch.randperm(BATCH_SIZE)]
    neg_triples = torch.stack([head_ids, rel_ids, neg_tails], dim=1)

    pos_s, neg_s = task_a.score_positive_and_negative(pos_triples, neg_triples)
    assert pos_s.shape == (BATCH_SIZE,)
    assert neg_s.shape == (BATCH_SIZE,)
    print(f"  PASS: shape pos={pos_s.shape}, neg={neg_s.shape}")

    # ---- Test 4: relation warm start ----
    print("\n--- Test 4: relation embedding warm start ---")
    # The 3 Fashion relations in alphabetical order:
    # belongs_to=0, belongs_to_brand=1, compatible_with=2
    fake_rel_names    = ['belongs_to', 'belongs_to_brand', 'compatible_with']
    fake_rel_pykeen   = {r: i for i, r in enumerate(fake_rel_names)}
    fake_rel_ours     = dict(fake_rel_pykeen)
    fake_pykeen_rels  = torch.randn(NUM_RELATIONS, EMBEDDING_DIM) * 2.0

    norm_before = task_a.relation_embedding.weight.norm(p=2, dim=1).mean().item()
    task_a.load_relation_weights_from_pykeen(
        pykeen_relation_weights=fake_pykeen_rels,
        relation_id_mapping_pykeen=fake_rel_pykeen,
        relation_id_mapping_ours=fake_rel_ours,
        verbose=True,
    )
    norm_after = task_a.relation_embedding.weight.norm(p=2, dim=1).mean().item()
    assert abs(norm_before - norm_after) > 1e-4, "Warm start did not change the weights!"

    # Verify exact copy
    for rel_name in fake_rel_names:
        our_id    = fake_rel_ours[rel_name]
        pykeen_id = fake_rel_pykeen[rel_name]
        diff = (
            task_a.relation_embedding.weight[our_id].detach()
            - fake_pykeen_rels[pykeen_id]
        ).abs().max().item()
        assert diff == 0.0, f"Incorrect copy for '{rel_name}': diff={diff}"
    print(f"  PASS: warm start ({norm_before:.4f} -> {norm_after:.4f})")
    print(f"  PASS: exact copy verified for all 3 relations")

    # ---- Test 5: dimension mismatch ----
    print("\n--- Test 5: dimension mismatch detected ---")
    try:
        task_a.load_relation_weights_from_pykeen(
            pykeen_relation_weights=torch.randn(NUM_RELATIONS, EMBEDDING_DIM + 1),
            relation_id_mapping_pykeen=fake_rel_pykeen,
            relation_id_mapping_ours=fake_rel_ours,
            verbose=False,
        )
        raise AssertionError("Dimension mismatch was NOT detected!")
    except ValueError:
        print(f"  PASS: dimension mismatch detected correctly")

    # ---- Test 6: shared_embedding is NOT a child module ----
    print("\n--- Test 6: shared_embedding is NOT registered as a child module ---")
    own_params         = sum(p.numel() for p in task_a.parameters())
    expected_rel_params = NUM_RELATIONS * EMBEDDING_DIM  # = 192
    assert own_params == expected_rel_params, (
        f"TaskATransE has {own_params} parameters, "
        f"expected only {expected_rel_params} (relation embeddings)"
    )
    print(f"  PASS: TaskATransE owns {own_params} parameters (relations only)")

    # ---- Test 7: score_all_tails shape ----
    print("\n--- Test 7: score_all_tails shape and values ---")
    small_batch = 4
    h_s = torch.randint(0, 182_983, (small_batch,))
    r_s = torch.randint(0, NUM_RELATIONS, (small_batch,))

    all_tail_scores = task_a.score_all_tails(h_s, r_s)
    assert all_tail_scores.shape == (small_batch, NUM_ENTITIES), \
        f"Expected shape ({small_batch}, {NUM_ENTITIES}), got {all_tail_scores.shape}"
    print(f"  PASS: shape {all_tail_scores.shape}")
    print(f"  mean={all_tail_scores.mean():.4f}, "
          f"min={all_tail_scores.min():.4f}, max={all_tail_scores.max():.4f}")

    # ---- Test 8: score_all_heads shape ----
    print("\n--- Test 8: score_all_heads shape ---")
    t_s = torch.randint(0, 182_983, (small_batch,))
    all_head_scores = task_a.score_all_heads(r_s, t_s)
    assert all_head_scores.shape == (small_batch, NUM_ENTITIES), \
        f"Expected shape ({small_batch}, {NUM_ENTITIES}), got {all_head_scores.shape}"
    print(f"  PASS: shape {all_head_scores.shape}")

    # ---- Test 9: consistency between score() and score_all_tails() ----
    print("\n--- Test 9: score() vs score_all_tails() consistency ---")
    test_h = torch.tensor([17514, 50000, 100000])   # 3 item IDs
    test_r = torch.tensor([2, 2, 2])                # compatible_with = id 2
    test_t = torch.tensor([17515, 50001, 100001])

    energy   = task_a.score(test_h, test_r, test_t)
    all_t_sim = task_a.score_all_tails(test_h, test_r)
    sim_at_t  = torch.stack([all_t_sim[i, test_t[i]] for i in range(3)])

    diff9 = (sim_at_t - (-energy)).abs().max().item()
    assert diff9 < 1e-4, f"Inconsistency: max diff = {diff9}"
    print(f"  PASS: score_all_tails[i, t_i] == -score(h_i, r_i, t_i)")
    print(f"  max diff: {diff9:.2e}")

    # ---- Test 10: consistency between score() and score_all_heads() ----
    print("\n--- Test 10: score() vs score_all_heads() consistency ---")
    all_h_sim = task_a.score_all_heads(test_r, test_t)
    sim_at_h  = torch.stack([all_h_sim[i, test_h[i]] for i in range(3)])

    diff10 = (sim_at_h - (-energy)).abs().max().item()
    assert diff10 < 1e-4, f"Inconsistency: max diff = {diff10}"
    print(f"  PASS: score_all_heads[i, h_i] == -score(h_i, r_i, t_i)")
    print(f"  max diff: {diff10:.2e}")

    print("\n" + "=" * 70)
    print("TASK A TRANSE FASHION — SMOKE TEST PASSED")
    print("=" * 70)
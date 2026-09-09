"""
losses_fashion.py
==================

Loss functions for Fashion alternate learning.

Task A (TransE): Margin Ranking Loss
    - identical to B2B: pos_score - neg_score + margin < 0
    - default margin = 5.19 (from Fashion TransE HPO, best_params)

Task B (SASRec): BPR Loss or CE Loss
    - BPR: default for alternate learning (as in B2B)
    - CE: available because it achieved the best performance
          in standalone Fashion SASRec HPO (trial 17, loss_type=CE)
    - The two losses are interchangeable in the alternate loop:
      simply pass loss_fn_b=SASRecCELoss() instead of SASRecBPRLoss()

Differences compared with B2B (losses.py):
    - SASRecCELoss added (not available in B2B)
    - Integration test uses JointAlternateModelFashion (not JointAlternateModel)
    - Corrected integration test: no projection layer is used (removed in Fashion)
    - default margin = 5.19 (same as B2B, confirmed by Fashion HPO)

File location:
    fashion_generalization/alternate_learning/models/losses_fashion.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================== #
# Task A — TransE Margin Ranking Loss
# =========================================================================== #

class TransELoss(nn.Module):
    """
    Margin Ranking Loss for Task A (TransE).

    TransE produces distances (energy): lower values = more plausible triples.
    We want: distance(pos) + margin < distance(neg)
    i.e.:    distance(pos) - distance(neg) + margin < 0

    Parameters
    ----------
    margin : float
        Loss margin. Default = 8.684 from Fashion TransE HPO.
        Typical of p_norm=1 (L1): with p_norm=2, typical values
        are lower (~1-2). Do not change without retraining TransE.
    """

    def __init__(self, margin: float = 8.684):
        super().__init__()
        self.margin = margin

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pos_scores : FloatTensor (B,)
            TransE distances for positive triples.
        neg_scores : FloatTensor (B,)
            TransE distances for negative triples (corrupted head or tail).

        Returns
        -------
        loss : scalar
        """
        return F.relu(pos_scores - neg_scores + self.margin).mean()


# =========================================================================== #
# Task B — SASRec BPR Loss
# =========================================================================== #

class SASRecBPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking Loss for Task B (SASRec).

    SASRec produces dot-product scores: higher values = more relevant items.
    We want: score(pos) > score(neg)

    Used as the default in alternate learning for consistency with B2B.
    Requires one negative sample for each positive sample (from the sampler).
    """

    def __init__(self):
        super().__init__()

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pos_scores : FloatTensor (B,)
            SASRec scores for target items (ground truth).
        neg_scores : FloatTensor (B,)
            SASRec scores for sampled negative items.

        Returns
        -------
        loss : scalar
        """
        return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()


# =========================================================================== #
# Task B — SASRec CE Loss
# =========================================================================== #

class SASRecCELoss(nn.Module):
    """
    Cross-Entropy Loss for Task B (SASRec).

    Alternative to BPR: instead of using a single negative for each positive,
    it computes CE over all catalog items (full softmax).
    This loss achieved the best performance in standalone Fashion SASRec HPO
    (trial 17, loss_type=CE, Recall@20=0.0964).

    In practice, the model must assign the highest probability to the target
    item compared with all other candidate items.

    Usage in the alternate loop:
        scores = model.score_items_sasrec(user_repr, candidate_ids)
        # scores shape: (B, N_items)
        # targets shape: (B,) — position of the target in the candidate array
        loss = ce_loss(scores, targets)

    Parameters
    ----------
    label_smoothing : float
        Label smoothing. Default = 0.0 (no smoothing,
        as in standalone RecBole training). Increase if overfitting occurs.
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        scores : FloatTensor (B, N)
            Scores over all N candidate items for each user in the batch.
        targets : LongTensor (B,)
            Index (along dimension N) of the target item for each user.
            IMPORTANT: this is the position in the candidate vector,
            NOT the unified ID. The dataloader must provide this index.

        Returns
        -------
        loss : scalar
        """
        return self.ce(scores, targets)


# =========================================================================== #
# SMOKE TEST
# =========================================================================== #

if __name__ == "__main__":
    import os
    import sys
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    if THIS_DIR not in sys.path:
        sys.path.insert(0, THIS_DIR)

    print("=" * 70)
    print("LOSSES FASHION — SMOKE TEST")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # TEST 1: TransELoss — direction and gradient
    # ------------------------------------------------------------------ #
    print("\n--- Test 1: TransELoss ---")
    transe_loss_fn = TransELoss(margin=5.19)

    pos = torch.tensor([1.0, 2.0, 0.5], requires_grad=True)
    neg = torch.tensor([8.0, 9.0, 7.0], requires_grad=True)

    # Ideal scenario: pos << neg, loss must be 0
    loss_ok = transe_loss_fn(pos, neg)
    assert loss_ok.item() == 0.0, \
        f"With pos<<neg the loss should be 0, got {loss_ok.item():.4f}"
    print(f"  Ideal scenario (pos<<neg): loss={loss_ok.item():.4f} ✓ (expected 0.0)")

    # Bad scenario: pos >> neg
    loss_bad = transe_loss_fn(neg, pos)
    assert loss_bad.item() > 0.0, \
        f"With pos>>neg the loss should be >0, got {loss_bad.item():.4f}"
    print(f"  Bad scenario (pos>>neg): loss={loss_bad.item():.4f} ✓ (expected >0)")

    assert loss_ok.item() < loss_bad.item(), "Incorrect TransELoss direction!"

    # Gradient
    loss_bad.backward()
    assert pos.grad is not None and pos.grad.abs().sum() > 0
    assert neg.grad is not None and neg.grad.abs().sum() > 0
    print(f"  Gradient: pos.grad={pos.grad.norm():.4f}, neg.grad={neg.grad.norm():.4f} ✓")
    print(f"  PASS: TransELoss")

    # ------------------------------------------------------------------ #
    # TEST 2: SASRecBPRLoss — direction and gradient
    # ------------------------------------------------------------------ #
    print("\n--- Test 2: SASRecBPRLoss ---")
    bpr_loss_fn = SASRecBPRLoss()

    pos_b = torch.tensor([5.0, 6.0, 4.0], requires_grad=True)
    neg_b = torch.tensor([1.0, 2.0, 0.5], requires_grad=True)

    # Ideal scenario: pos >> neg, loss must be low
    loss_ok_b = bpr_loss_fn(pos_b, neg_b)
    print(f"  Ideal scenario (pos>>neg): loss={loss_ok_b.item():.4f} ✓ (expected low)")

    # Bad scenario: pos << neg
    loss_bad_b = bpr_loss_fn(neg_b, pos_b)
    print(f"  Bad scenario (pos<<neg): loss={loss_bad_b.item():.4f} ✓ (expected high)")

    assert loss_ok_b.item() < loss_bad_b.item(), "Incorrect BPR Loss direction!"

    loss_bad_b.backward()
    assert pos_b.grad is not None and pos_b.grad.abs().sum() > 0
    assert neg_b.grad is not None and neg_b.grad.abs().sum() > 0
    print(f"  Gradient: pos.grad={pos_b.grad.norm():.4f}, neg.grad={neg_b.grad.norm():.4f} ✓")
    print(f"  PASS: SASRecBPRLoss")

    # ------------------------------------------------------------------ #
    # TEST 3: SASRecCELoss — direction and gradient
    # ------------------------------------------------------------------ #
    print("\n--- Test 3: SASRecCELoss ---")
    ce_loss_fn = SASRecCELoss()

    # scores (B=3, N=5 candidate items), targets = target index
    scores_ce = torch.tensor([
        [0.1, 0.2, 5.0, 0.1, 0.1],   # target is at position 2 → score 5.0 >> others
        [0.1, 0.1, 0.1, 6.0, 0.1],   # target is at position 3
        [7.0, 0.1, 0.1, 0.1, 0.1],   # target is at position 0
    ], requires_grad=True)
    targets_ce = torch.tensor([2, 3, 0])

    loss_ok_ce = ce_loss_fn(scores_ce, targets_ce)
    print(f"  Ideal scenario (target has max score): loss={loss_ok_ce.item():.4f} ✓ (expected low)")

    # Bad scenario: target has minimum score
    scores_bad_ce = torch.tensor([
        [5.0, 5.0, 0.1, 5.0, 5.0],
        [5.0, 5.0, 5.0, 0.1, 5.0],
        [0.1, 5.0, 5.0, 5.0, 5.0],
    ], requires_grad=True)
    loss_bad_ce = ce_loss_fn(scores_bad_ce, targets_ce)
    print(f"  Bad scenario (target has min score): loss={loss_bad_ce.item():.4f} ✓ (expected high)")

    assert loss_ok_ce.item() < loss_bad_ce.item(), "Incorrect CE Loss direction!"

    loss_bad_ce.backward()
    assert scores_bad_ce.grad is not None
    print(f"  Gradient: scores.grad norm={scores_bad_ce.grad.norm():.4f} ✓")
    print(f"  PASS: SASRecCELoss")

    # ------------------------------------------------------------------ #
    # TEST 4: integration with JointAlternateModelFashion
    # ------------------------------------------------------------------ #
    print("\n--- Test 4: integration with JointAlternateModelFashion ---")
    try:
        from joint_model_fashion import JointAlternateModelFashion

        NUM_ENTITIES    = 182_984
        NUM_RELATIONS   = 3
        KGE_DIM         = 64
        UNIFIED_PADDING = 182_983
        ITEM_ID_START   = 17_514
        NUM_ITEMS_REAL  = 165_469
        BATCH           = 4
        SEQ_LEN         = 50

        model = JointAlternateModelFashion(
            num_entities=NUM_ENTITIES,
            num_relations=NUM_RELATIONS,
            kge_dim=KGE_DIM,
            unified_padding_idx=UNIFIED_PADDING,
        )

        # ---- Task A: TransELoss ----
        h = torch.randint(ITEM_ID_START, NUM_ENTITIES - 1, (BATCH,))
        r = torch.zeros(BATCH, dtype=torch.long)
        t_pos = torch.randint(ITEM_ID_START, NUM_ENTITIES - 1, (BATCH,))
        t_neg = torch.randint(ITEM_ID_START, NUM_ENTITIES - 1, (BATCH,))

        pos_scores_a = model.forward_kge(h, r, t_pos)
        neg_scores_a = model.forward_kge(h, r, t_neg)
        loss_a = transe_loss_fn(pos_scores_a, neg_scores_a)

        model.zero_grad()
        loss_a.backward()

        grad_a = model.shared_embedding.embedding.weight.grad
        assert grad_a is not None and grad_a.abs().sum().item() > 0, \
            "No gradient on SharedEmbedding from Task A!"
        assert grad_a[UNIFIED_PADDING].abs().sum().item() == 0.0, \
            "Padding row received gradient from Task A!"
        print(f"  Task A + TransELoss: SharedEmbedding grad={grad_a.abs().sum():.4f} ✓")

        # ---- Task B: BPR Loss ----
        item_seq = torch.full((BATCH, SEQ_LEN), UNIFIED_PADDING, dtype=torch.long)
        for b in range(BATCH):
            sl = torch.randint(2, SEQ_LEN, (1,)).item()
            item_seq[b, :sl] = torch.randint(ITEM_ID_START, NUM_ENTITIES - 1, (sl,))
        item_seq_len = (item_seq != UNIFIED_PADDING).sum(dim=1)

        user_repr = model.forward_sasrec(item_seq, item_seq_len)

        pos_ids = torch.randint(ITEM_ID_START, NUM_ENTITIES - 1, (BATCH,))
        neg_ids = torch.randint(ITEM_ID_START, NUM_ENTITIES - 1, (BATCH,))

        # score_items_sasrec takes a candidate vector (N,)
        # for BPR we need separate scalar scores for positive and negative items
        pos_scores_b = model.score_items_sasrec(
            user_repr, pos_ids
        ).gather(1, torch.zeros(BATCH, 1, dtype=torch.long)).squeeze(1)

        # Simpler: direct dot product
        pos_emb = model.shared_embedding(pos_ids)   # (B, 64)
        neg_emb = model.shared_embedding(neg_ids)   # (B, 64)
        pos_scores_b = (user_repr * pos_emb).sum(dim=-1)  # (B,)
        neg_scores_b = (user_repr * neg_emb).sum(dim=-1)  # (B,)

        loss_b_bpr = bpr_loss_fn(pos_scores_b, neg_scores_b)

        model.zero_grad()
        loss_b_bpr.backward()

        grad_b = model.shared_embedding.embedding.weight.grad
        assert grad_b is not None and grad_b.abs().sum().item() > 0, \
            "No gradient on SharedEmbedding from Task B BPR!"
        assert grad_b[UNIFIED_PADDING].abs().sum().item() == 0.0, \
            "Padding row received gradient from Task B BPR!"
        print(f"  Task B + BPRLoss:   SharedEmbedding grad={grad_b.abs().sum():.4f} ✓")

        # ---- Task B: CE Loss ----
        candidate_ids = torch.arange(ITEM_ID_START, NUM_ENTITIES - 1, dtype=torch.long)
        user_repr2 = model.forward_sasrec(item_seq, item_seq_len)
        scores_all = model.score_items_sasrec(user_repr2, candidate_ids)  # (B, 165469)

        # target: index in the candidate array (pos_ids - ITEM_ID_START)
        targets_ce_real = (pos_ids - ITEM_ID_START)  # (B,)

        loss_b_ce = ce_loss_fn(scores_all, targets_ce_real)

        model.zero_grad()
        loss_b_ce.backward()

        grad_ce = model.shared_embedding.embedding.weight.grad
        assert grad_ce is not None and grad_ce.abs().sum().item() > 0, \
            "No gradient on SharedEmbedding from Task B CE!"
        assert grad_ce[UNIFIED_PADDING].abs().sum().item() == 0.0, \
            "Padding row received gradient from Task B CE!"
        print(f"  Task B + CELoss:    SharedEmbedding grad={grad_ce.abs().sum():.4f} ✓")

        print(f"  PASS: integration with JointAlternateModelFashion")

    except ImportError as e:
        print(f"  SKIPPED: {e}")

    print("\n" + "=" * 70)
    print("LOSSES FASHION — SMOKE TEST PASSED")
    print("=" * 70)
    print()
    print("Available losses:")
    print("  TransELoss      : margin=8.684 (p_norm=1, from Fashion HPO)")
    print("  SASRecBPRLoss   : default for alternate learning (consistent with B2B)")
    print("  SASRecCELoss    : alternative — better in standalone Fashion HPO")
    print()
    print("Usage during training:")
    print("  loss_fn_a = TransELoss(margin=8.684)")
    print("  loss_fn_b = SASRecBPRLoss()   # or SASRecCELoss()")
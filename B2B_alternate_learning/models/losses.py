import torch
import torch.nn as nn
import torch.nn.functional as F

class TransELoss(nn.Module):
    """
    Margin Ranking Loss per il Task A (KGE - TransE).
     TransE gives scores based on distances (lower is better).
    we want: pos_score + margin < neg_score.
    That is: pos_score - neg_score + margin < 0.
    """
    def __init__(self, margin: float = 5.19): # 5.19 from HPO Step 2, best_paramsTransE.json
        super().__init__()
        self.margin = margin

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """
        Calculate the margin loss.
        pos_scores and neg_scores must have the same shape (e.g. [batch_size]).
        """
        # torch.relu implementa la logica max(0, x)
        loss = F.relu(pos_scores - neg_scores + self.margin)
        return loss.mean()


class SASRecBPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking (BPR) Loss per il Task B (SASRec).
    SASRec restituisce score basati su dot product (valori più alti = item migliori).
    Vogliamo massimizzare la distanza tra lo score positivo e quello negativo.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """
        compute the BPR loss given positive and negative scores.
        pos_scores and neg_scores must have the same shape (e.g. [batch_size]).
        """
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8)
        return loss.mean()


# =========================================================================== #
# SMOKE TEST
# =========================================================================== #

if __name__ == "__main__":
    print("=" * 70)
    print("LOSSES SMOKE TEST")
    print("=" * 70)

    # Tensor dummy with requirements_grad to test gradient flow
    # They simulate the network output (e.g. embedding matrices)
    dummy_pos = torch.tensor([1.0, 2.0, 0.5], requires_grad=True)
    dummy_neg = torch.tensor([5.0, 6.0, 4.0], requires_grad=True)

    # ---------------------------------------------------------
    # TEST TASK A: TRANSE LOSS (Distances, smaller is better)
    # ---------------------------------------------------------
    print("\n--- Test Task A: TransE Margin Loss ---")
    transe_loss_fn = TransELoss(margin=2.0)
    # 1. Ideal Scenario: Positives have much less distance than negatives
    loss_ideale_A = transe_loss_fn(dummy_pos, dummy_neg)
    print(f"  Ideal loss (pos < neg): {loss_ideale_A.item():.4f} (expected ~ 0.0)")
    # 2. Terrible Scenario: Positives have greater distance than negatives
    loss_pessima_A = transe_loss_fn(dummy_neg, dummy_pos)
    print(f"  worst loss (pos > neg): {loss_pessima_A.item():.4f} (expected > 0.0)")

    assert loss_ideale_A.item() < loss_pessima_A.item(), "DIRECTION ERROR: TransE Loss does not penalize correctly!"

    # 3. Gradient Flow
    dummy_pos.grad = None
    dummy_neg.grad = None
    loss_pessima_A.backward()
    assert dummy_pos.grad is not None and dummy_neg.grad is not None, "Gradients do not flow in TransE Loss!"
    print(f"  Gradients flow: TRANSE PASS (dummy_pos grad: {dummy_pos.grad.norm().item():.3f})")

    # ---------------------------------------------------------
    # TEST TASK B: SASREC BPR LOSS (Dot Product, higher is better)
    # ---------------------------------------------------------
    print("\n--- Test Task B: SASRec BPR Loss ---")
    bpr_loss_fn = SASRecBPRLoss()

    # Create new clean tensors without previous gradients
    # For SASRec: high scores are better
    sasrec_pos = torch.tensor([5.0, 6.0, 4.0], requires_grad=True)
    sasrec_neg = torch.tensor([1.0, 2.0, 0.5], requires_grad=True)

    # 1. Ideal Scenario: Positives have score much higher than negatives
    loss_ideale_B = bpr_loss_fn(sasrec_pos, sasrec_neg)
    print(f"  Ideal loss (pos > neg): {loss_ideale_B.item():.4f} (expected low, close to zero)")

    # 2. Terrible Scenario: Positives have score much lower than negatives
    loss_pessima_B = bpr_loss_fn(sasrec_neg, sasrec_pos)
    print(f"  worst loss (pos < neg): {loss_pessima_B.item():.4f} (expected high)")

    assert loss_ideale_B.item() < loss_pessima_B.item(), "DIRECTION ERROR: BPR Loss does not penalize correctly!"

    # 3. Gradient Flow
    sasrec_pos.grad = None
    sasrec_neg.grad = None
    loss_pessima_B.backward()
    assert sasrec_pos.grad is not None and sasrec_neg.grad is not None, "Gradients do not flow in BPR Loss!"
    print(f"  Gradients flow: BPR PASS (sasrec_pos grad: {sasrec_pos.grad.norm().item():.3f})")

    # ---------------------------------------------------------
    # JOINT MODEL INTEGRATION TESTING (VERIFY REAL FLOW)
    # ---------------------------------------------------------
    print("\n--- Joint Model Integration Testing Real Flow ---")
    try:
        from joint_model import JointAlternateModel

        # Set up a small joint model with dummy parameters
        NUM_ENTITIES = 1000
        NUM_RELATIONS = 5
        NUM_ITEMS = 500
        BATCH = 4

        model = JointAlternateModel(
            num_entities=NUM_ENTITIES, num_relations=NUM_RELATIONS, num_items=NUM_ITEMS,
            kge_dim=32, sasrec_dim=16
        )

        # == INTEGRATION A: TransE ==
        h_pos = torch.randint(0, NUM_ENTITIES, (BATCH,))
        r_pos = torch.randint(0, NUM_RELATIONS, (BATCH,))
        t_pos = torch.randint(0, NUM_ENTITIES, (BATCH,))
        t_neg = torch.randint(0, NUM_ENTITIES, (BATCH,))
        pos_scores_A = model.forward_kge(h_pos, r_pos, t_pos)
        neg_scores_A = model.forward_kge(h_pos, r_pos, t_neg)
        loss_A = transe_loss_fn(pos_scores_A, neg_scores_A)
        model.zero_grad()
        loss_A.backward()
        grad_shared_A = model.shared_embedding.embedding.weight.grad
        assert grad_shared_A is not None and grad_shared_A.abs().sum().item() > 0, "No grad on SharedEmbedding from Task A!"
        print("  Integration Task A: PASS (The gradients reach the SharedEmbedding)")

        # == INTEGRATION B: SASRec ==
        item_seq = torch.randint(1, NUM_ENTITIES, (BATCH, 10))
        item_seq_len = torch.randint(1, 11, (BATCH,))
        from task_b_sasrec import TaskBSASRec
        # Let's pretend the samplers pass us positive and negative items
        pos_items = torch.randint(1, NUM_ENTITIES, (BATCH,))
        neg_items = torch.randint(1, NUM_ENTITIES, (BATCH,))
        user_repr = model.forward_sasrec(item_seq, item_seq_len)
        # Exact logic you will then use in training (dot product between user and item projection)
        pos_emb = model.task_b.projection(model.shared_embedding(pos_items))
        neg_emb = model.task_b.projection(model.shared_embedding(neg_items))
        scores_pos_B = (user_repr * pos_emb).sum(dim=-1)
        scores_neg_B = (user_repr * neg_emb).sum(dim=-1)
        loss_B = bpr_loss_fn(scores_pos_B, scores_neg_B)
        model.zero_grad()
        loss_B.backward()
        grad_shared_B = model.shared_embedding.embedding.weight.grad
        grad_proj_B = model.task_b.projection.weight.grad

        # Check that the gradients have been computed and are not zero
        assert grad_shared_B is not None and grad_shared_B.abs().sum().item() > 0, "No grad on SharedEmbedding from Task B!"
        assert grad_proj_B is not None and grad_proj_B.abs().sum().item() > 0, "No grad on Projection from Task B!"
        print("  Integration Task B: PASS (The gradients reach the SharedEmbedding and the Projection Layer)")

    except ImportError:
        print("  SKIPPED: joint_model.py not found in the same folder (this is fine if you are testing losses.py in isolation).")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY! :)")
    print("=" * 70)
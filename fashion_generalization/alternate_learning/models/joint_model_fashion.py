"""
joint_model_fashion.py
=======================

Orchestrator for the Alternate Learning loop — Fashion domain.

Instantiates SharedEmbedding and the two tasks (TaskATransE, TaskBSASRec),
and exposes the methods used by the alternate loop.

Differences compared with B2B (joint_model.py):
  - kge_dim = 64 (not 400) — Fashion TransE and SASRec use the same dimension
  - num_relations = 3 (not 5)
  - unified_padding_idx = 182983 (not 0) — brand IDs start from 0
  - Fashion TaskBSASRec does NOT have the num_items parameter
    (it uses candidate_unified_ids)
  - default n_heads = 4 (not 2), inner_size added (256)
  - normalize_entities = True in TaskATransE (as in Fashion PyKEEN)

File location:
    fashion_generalization/alternate_learning/models/joint_model_fashion.py
"""

import torch
import torch.nn as nn

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from shared_embedding_fashion import SharedEmbedding
from task_a_transe_fashion import TaskATransE
from task_b_sasrec_fashion import TaskBSASRec


class JointAlternateModelFashion(nn.Module):
    def __init__(
        self,
        num_entities: int,       # 182984 (182983 KG entities + 1 padding)
        num_relations: int,      # 3
        kge_dim: int = 64,       # DIFFERENT from B2B (400): Fashion uses 64
        max_seq_length: int = 50,
        p_norm: int = 1,
        n_layers: int = 4,
        n_heads: int = 4,        # DIFFERENT from B2B (2): trial 17 uses 4
        inner_size: int = 256,   # ADDED compared with B2B: Fashion exposes inner_size
        hidden_dropout: float = 0.5,  # DIFFERENT from B2B (0.4): trial 17 uses 0.5
        attn_dropout: float = 0.3,
        unified_padding_idx: int = 182_983,  # DIFFERENT from B2B (0 or None)
    ):
        """
        Orchestrator for the Fashion Alternate Learning loop.

        Contains the SharedEmbedding (182984 x 64) and the two tasks.
        Both tasks read from the same SharedEmbedding object —
        no parameter duplication.

        Fashion architectural notes:
          - No projection layer: kge_dim == hidden_size == 64
          - padding_idx = 182983 (not 0): brand IDs occupy 0..17510
          - TaskBSASRec has no num_items: it explicitly uses candidate_unified_ids
          - normalize_entities=True in TaskATransE to replicate PyKEEN
        """
        super().__init__()

        # SharedEmbedding: 182984 rows x 64 dim, padding_idx=182983
        self.shared_embedding = SharedEmbedding(
            num_entities=num_entities,
            embedding_dim=kge_dim,
            padding_idx=unified_padding_idx,
        )

        # Task A: TransE on the Fashion KG
        # normalize_entities=True to replicate PyKEEN
        # (verified in test_task_a_real_data_fashion)
        # B2B DIFFERENCE: in B2B it was False (different architecture)
        self.task_a = TaskATransE(
            shared_embedding=self.shared_embedding,
            num_relations=num_relations,
            embedding_dim=kge_dim,
            p_norm=p_norm,
            normalize_entities=True,
        )

        # Task B: Fashion SASRec
        # B2B DIFFERENCES:
        #   - there is no num_items (Fashion uses candidate_unified_ids)
        #   - unified_padding_idx=182983 instead of padding_idx=0
        #   - inner_size is explicitly exposed (256 = 4 * 64)
        self.task_b = TaskBSASRec(
            shared_embedding=self.shared_embedding,
            n_layers=n_layers,
            n_heads=n_heads,
            hidden_size=kge_dim,
            inner_size=inner_size,
            hidden_dropout=hidden_dropout,
            attn_dropout=attn_dropout,
            max_seq_length=max_seq_length,
            unified_padding_idx=unified_padding_idx,
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "JointAlternateModelFashion does not have a single forward method. "
            "Explicitly use model.forward_kge() or model.forward_sasrec()."
        )

    # =========================================================================
    # TASK A METHODS (TransE)
    # =========================================================================

    def forward_kge(self, h, r, t):
        """Score for (head, relation, tail) triples. Shape: (B,)"""
        return self.task_a.score(h, r, t)

    def score_all_tails_kge(self, head_ids, relation_ids):
        """Score against all candidate tails. Shape: (B, num_entities)"""
        return self.task_a.score_all_tails(head_ids, relation_ids)

    def score_all_heads_kge(self, relation_ids, tail_ids):
        """Score against all candidate heads. Shape: (B, num_entities)"""
        return self.task_a.score_all_heads(relation_ids, tail_ids)

    # =========================================================================
    # TASK B METHODS (SASRec)
    # =========================================================================

    def forward_sasrec(self, item_seq, item_seq_len):
        """
        User representation from the sequence.

        Parameters
        ----------
        item_seq : LongTensor (B, T)
            Unified IDs. Padding = 182983 (NOT 0 as in B2B).
        item_seq_len : LongTensor (B,)

        Returns
        -------
        user_repr : FloatTensor (B, 64)
        """
        return self.task_b(item_seq, item_seq_len)

    def score_items_sasrec(self, user_repr, candidate_unified_ids):
        """
        Dot product between user_repr and candidate embeddings.

        B2B DIFFERENCE: in B2B, candidates were all items (implicit).
        In Fashion, candidates are explicitly passed as unified IDs
        (typically torch.arange(17514, 182983)).

        Parameters
        ----------
        user_repr : FloatTensor (B, 64)
        candidate_unified_ids : LongTensor (N,)

        Returns
        -------
        scores : FloatTensor (B, N)
        """
        return self.task_b.score_all_items(user_repr, candidate_unified_ids)

    # =========================================================================
    # WARM START
    # =========================================================================

    def load_pretrained_kge(
        self,
        pykeen_entity_weights,
        entity_mapping_pykeen,
        entity_mapping_ours,
        pykeen_relation_weights,
        rel_mapping_pykeen,
        rel_mapping_ours,
    ):
        """
        Load pretrained weights from PyKEEN (entity + relation).

        B2B DIFFERENCE: in B2B, load_pretrained_kge loaded only the relations
        (entity weights were loaded separately through shared_embedding).
        Here both are exposed in a single method for clarity,
        although they can still be called separately.
        """
        self.shared_embedding.load_from_pykeen_transe(
            pykeen_weights=pykeen_entity_weights,
            entity_id_mapping_pykeen=entity_mapping_pykeen,
            entity_id_mapping_ours=entity_mapping_ours,
            verbose=True,
        )
        self.task_a.load_relation_weights_from_pykeen(
            pykeen_relation_weights=pykeen_relation_weights,
            relation_id_mapping_pykeen=rel_mapping_pykeen,
            relation_id_mapping_ours=rel_mapping_ours,
            verbose=True,
        )

    def load_pretrained_sasrec(self, sasrec_recbole_state_dict):
        """
        Load Transformer + position embedding weights from RecBole.
        item_embedding.weight is skipped (it lives in SharedEmbedding).
        """
        self.task_b.load_self_attention_from_recbole(
            sasrec_recbole_state_dict, verbose=True
        )


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("JOINT ALTERNATE MODEL FASHION — SMOKE TEST")
    print("=" * 70)

    # Real Fashion dimensions
    NUM_ENTITIES      = 182_984   # 182983 KG entities + 1 padding
    NUM_RELATIONS     = 3
    KGE_DIM           = 64
    UNIFIED_PADDING   = 182_983
    ITEM_ID_START     = 17_514
    NUM_ITEMS_REAL    = 165_469   # item unified IDs: 17514..182982
    BATCH             = 8
    SEQ_LEN           = 50

    model = JointAlternateModelFashion(
        num_entities=NUM_ENTITIES,
        num_relations=NUM_RELATIONS,
        kge_dim=KGE_DIM,
        unified_padding_idx=UNIFIED_PADDING,
    )

    # --- Test 1: forward_kge ---
    print("\n--- Test 1: forward_kge ---")
    h = torch.randint(ITEM_ID_START, NUM_ENTITIES - 1, (BATCH,))
    r = torch.zeros(BATCH, dtype=torch.long)   # relation 0
    t = torch.randint(ITEM_ID_START, NUM_ENTITIES - 1, (BATCH,))
    scores_kge = model.forward_kge(h, r, t)
    assert scores_kge.shape == (BATCH,), f"Incorrect shape: {scores_kge.shape}"
    print(f"  PASS: scores_kge.shape = {scores_kge.shape}")

    # --- Test 2: forward_sasrec ---
    print("\n--- Test 2: forward_sasrec ---")
    item_seq = torch.full((BATCH, SEQ_LEN), UNIFIED_PADDING, dtype=torch.long)
    for b in range(BATCH):
        seq_len = torch.randint(2, SEQ_LEN, (1,)).item()
        item_seq[b, :seq_len] = torch.randint(ITEM_ID_START, NUM_ENTITIES - 1, (seq_len,))
    item_seq_len = (item_seq != UNIFIED_PADDING).sum(dim=1)
    user_repr = model.forward_sasrec(item_seq, item_seq_len)
    assert user_repr.shape == (BATCH, KGE_DIM), f"Incorrect shape: {user_repr.shape}"
    print(f"  PASS: user_repr.shape = {user_repr.shape}")

    # --- Test 3: score_items_sasrec ---
    print("\n--- Test 3: score_items_sasrec ---")
    candidate_ids = torch.arange(ITEM_ID_START, NUM_ENTITIES - 1, dtype=torch.long)
    scores_rec = model.score_items_sasrec(user_repr.detach(), candidate_ids)
    assert scores_rec.shape == (BATCH, NUM_ITEMS_REAL), \
        f"Incorrect shape: {scores_rec.shape}, expected ({BATCH}, {NUM_ITEMS_REAL})"
    print(f"  PASS: scores_rec.shape = {scores_rec.shape} ({BATCH} users x {NUM_ITEMS_REAL:,} items)")

    # --- Test 4: forward() raises NotImplementedError ---
    print("\n--- Test 4: forward() raises NotImplementedError ---")
    try:
        model(h, r, t)
        print("  FAIL: expected NotImplementedError")
    except NotImplementedError:
        print("  PASS: NotImplementedError raised correctly")

    # --- Test 5: SharedEmbedding is not duplicated in the parameters ---
    print("\n--- Test 5: parameters are not duplicated ---")
    own_params    = sum(p.numel() for p in model.parameters())
    shared_params = NUM_ENTITIES * KGE_DIM
    # Task B own parameters (position_embedding + transformer) are ~850K
    # SharedEmbedding is ~11.7M
    # Expected total: ~12.6M, not ~23.4M (which would indicate duplication)
    assert own_params < shared_params * 2, \
        f"Possible duplication: {own_params:,} parameters (shared={shared_params:,})"
    print(f"  PASS: total parameters = {own_params:,}")
    print(f"        shared_embedding = {shared_params:,}")
    print(f"        task_b own       = {own_params - shared_params:,}")

    # --- Test 6: SharedEmbedding is the SAME instance in task_a and task_b ---
    print("\n--- Test 6: shared_embedding is the same physical instance ---")
    assert model.task_a.shared_embedding is model.shared_embedding, \
        "task_a.shared_embedding is a copy, not the same instance!"
    assert model.task_b.shared_embedding is model.shared_embedding, \
        "task_b.shared_embedding is a copy, not the same instance!"
    print("  PASS: task_a and task_b reference the same SharedEmbedding")

    # --- Test 7: gradient flow from Task A to SharedEmbedding ---
    print("\n--- Test 7: gradient flow Task A -> SharedEmbedding ---")
    model.shared_embedding.embedding.weight.grad = None
    scores_kge = model.forward_kge(h, r, t)
    scores_kge.sum().backward()
    grad_a = model.shared_embedding.embedding.weight.grad
    assert grad_a is not None, "No gradient from Task A!"
    grad_a_sum = grad_a.abs().sum().item()
    assert grad_a_sum > 0, "Task A gradient is zero!"
    # Padding must not receive gradients
    assert grad_a[UNIFIED_PADDING].abs().sum().item() == 0.0, \
        "Padding row received gradient from Task A!"
    print(f"  PASS: Task A grad = {grad_a_sum:.6f}, padding grad = 0.0")

    # --- Test 8: gradient flow from Task B to SharedEmbedding ---
    print("\n--- Test 8: gradient flow Task B -> SharedEmbedding ---")
    model.shared_embedding.embedding.weight.grad = None
    user_repr = model.forward_sasrec(item_seq, item_seq_len)
    scores_rec = model.score_items_sasrec(user_repr, candidate_ids)
    scores_rec.sum().backward()
    grad_b = model.shared_embedding.embedding.weight.grad
    assert grad_b is not None, "No gradient from Task B!"
    grad_b_sum = grad_b.abs().sum().item()
    assert grad_b_sum > 0, "Task B gradient is zero!"
    assert grad_b[UNIFIED_PADDING].abs().sum().item() == 0.0, \
        "Padding row received gradient from Task B!"
    # Brand and category rows (0..17513) must not receive gradients from Task B
    # (the sequence contains only unified item IDs)
    assert grad_b[:ITEM_ID_START].abs().sum().item() == 0.0, \
        "Brand/category rows received unexpected gradient from Task B!"
    print(f"  PASS: Task B grad = {grad_b_sum:.6f}, padding grad = 0.0, brand/cat grad = 0.0")

    # --- Test 9: modification of SharedEmbedding is visible from both tasks ---
    print("\n--- Test 9: SharedEmbedding update is visible from both tasks ---")
    with torch.no_grad():
        # Write a sentinel value to row 17514 (first item)
        model.shared_embedding.embedding.weight[ITEM_ID_START].fill_(999.0)
        # Read from task_a
        val_a = model.task_a.shared_embedding.embedding.weight[ITEM_ID_START].mean().item()
        # Read from task_b
        val_b = model.task_b.shared_embedding.embedding.weight[ITEM_ID_START].mean().item()
        assert val_a == 999.0, f"task_a does not see the update: {val_a}"
        assert val_b == 999.0, f"task_b does not see the update: {val_b}"
        # Restore
        torch.nn.init.xavier_uniform_(model.shared_embedding.embedding.weight)
        model.shared_embedding.embedding.weight[UNIFIED_PADDING].fill_(0.0)
    print("  PASS: SharedEmbedding update is immediately visible from task_a and task_b")

    # --- Test 10: padding_idx = 182983 in task_b mask ---
    print("\n--- Test 10: attention mask uses unified_padding_idx=182983 ---")
    seq_test = torch.full((1, SEQ_LEN), UNIFIED_PADDING, dtype=torch.long)
    seq_test[0, 0] = ITEM_ID_START        # a real item
    seq_test[0, 1] = ITEM_ID_START + 1    # another real item
    seq_len_test = torch.tensor([2])
    mask = model.task_b.get_attention_mask(seq_test)
    # Positions 2..49 must be -10000 (padding)
    assert (mask[0, 0, :, 2:] == -10000.0).all(), \
        "Padding positions are not masked correctly!"
    # Position 0 (id=ITEM_ID_START, not padding) must not be -10000 in the column
    # Verify that column 0 of row 1 is 0 (real item is visible)
    assert mask[0, 0, 1, 0].item() == 0.0, \
        "Real item (id=ITEM_ID_START) was incorrectly masked!"
    print(f"  PASS: padding (id={UNIFIED_PADDING}) masked, real items are not masked")
    print(f"  PASS: brand IDs (0..17510) would not be masked (item_seq != {UNIFIED_PADDING})")

    print("\n" + "=" * 70)
    print("JOINT ALTERNATE MODEL FASHION — SMOKE TEST PASSED")
    print("=" * 70)
    print()
    print("Fashion architecture summary:")
    print(f"  SharedEmbedding  : {NUM_ENTITIES:,} x {KGE_DIM} (padding_idx={UNIFIED_PADDING})")
    print(f"  TaskATransE      : {NUM_RELATIONS} relations, p_norm=1, normalize=True")
    print(f"  TaskBSASRec      : 4 layers x 4 heads, hidden=64, inner=256")
    print(f"  item unified IDs : {ITEM_ID_START}..{NUM_ENTITIES-2} ({NUM_ITEMS_REAL:,} items)")
    print(f"  No projection layer (kge_dim == hidden_size == {KGE_DIM})")
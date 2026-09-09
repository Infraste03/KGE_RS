import torch
import torch.nn as nn

from shared_embedding import SharedEmbedding
from task_a_transe import TaskATransE
from task_b_sasrec import TaskBSASRec

class JointAlternateModel(nn.Module):

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        num_items: int,
        kge_dim: int = 400,
        max_seq_length: int = 50,
        p_norm: int = 1,
        n_layers: int = 4,
        n_heads: int = 2,
        hidden_dropout: float = 0.4,
        attn_dropout: float = 0.3,
        padding_idx: int = None
    ):
        """
        Orchestrator for the Alternate Learning loop.
        Contains the shared embedding and the two tasks (TransE and SASRec).
        Both work natively in 400-dim space (kge_dim).
        """
        super().__init__()

        self.shared_embedding = SharedEmbedding(
            num_entities=num_entities,
            embedding_dim=kge_dim,
            padding_idx=padding_idx
        )
        self.task_a = TaskATransE(
            shared_embedding=self.shared_embedding,
            num_relations=num_relations,
            embedding_dim=kge_dim,
            p_norm=p_norm,
            normalize_entities=False
        )

        self.task_b = TaskBSASRec(
            shared_embedding=self.shared_embedding,
            num_items=num_items,
            hidden_size=kge_dim,
            max_seq_length=max_seq_length,
            n_layers=n_layers,
            n_heads=n_heads,
            hidden_dropout=hidden_dropout,
            attn_dropout=attn_dropout,
            padding_idx=padding_idx
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "JointAlternateModel non ha un forward unico. "
            "Usa esplicitamente model.forward_kge() o model.forward_sasrec()."
        )

    # ==========================================
    # METHODS FOR TASK A (KGE)
    # ==========================================

    def forward_kge(self, h, r, t):
        return self.task_a.score(h, r, t)

    def score_all_tails_kge(self, head_ids, relation_ids):
        return self.task_a.score_all_tails(head_ids, relation_ids)

    def score_all_heads_kge(self, relation_ids, tail_ids):
        return self.task_a.score_all_heads(relation_ids, tail_ids)

    # ==========================================
    # METHODS FOR TASK B (SASREC)
    # ==========================================
    def forward_sasrec(self, item_seq, item_seq_len):
        """
       Compute the user representation from the item sequence using the SASRec model.
       The output is a 400-dimensional vector (kge_dim) without any projection.
        """
        return self.task_b(item_seq, item_seq_len)

    def score_items_sasrec(self, user_repr, candidate_unified_ids):
        """
        Score candidate items for a given user representation using the SASRec model.
        The user_repr is expected to be a 400-dimensional vector (kge_dim).
        """
        return self.task_b.score_all_items(user_repr, candidate_unified_ids)

    # ==========================================
    # utility methods / WARM START
    # ==========================================

    def load_pretrained_kge(self, pykeen_relations, rel_mapping_pykeen, rel_mapping_ours):
        """
        Load the pretrained relation embeddings from a PyKEEN TransE model (phase 2).
        """
        self.task_a.load_relation_weights_from_pykeen(
            pykeen_relation_weights=pykeen_relations,
            relation_id_mapping_pykeen=rel_mapping_pykeen,
            relation_id_mapping_ours=rel_mapping_ours,
            verbose=True
        )

    def load_pretrained_sasrec(self, sasrec_recbole_state_dict):
        """
        Load the pretrained self-attention weights from a RecBole model (phase 3).
        """
        self.task_b.load_self_attention_from_recbole(sasrec_recbole_state_dict, verbose=True)

if __name__ == "__main__":
    print("=" * 70)
    print("JOINT ALTERNATE MODEL SMOKE TEST")
    print("=" * 70)

    NUM_ENTITIES = 24_957
    NUM_RELATIONS = 5
    NUM_ITEMS = 21_135
    KGE_DIM = 400
    BATCH = 8

    model = JointAlternateModel(
        num_entities=NUM_ENTITIES,
        num_relations=NUM_RELATIONS,
        num_items=NUM_ITEMS,
        kge_dim=KGE_DIM
    )

    # --- Test 1: forward_kge ---
    print("\n--- Test 1: forward_kge ---")
    h = torch.randint(0, NUM_ENTITIES, (BATCH,))
    r = torch.randint(0, NUM_RELATIONS, (BATCH,))
    t = torch.randint(0, NUM_ENTITIES, (BATCH,))
    scores_kge = model.forward_kge(h, r, t)
    assert scores_kge.shape == (BATCH,), f"Expected ({BATCH},), got {scores_kge.shape}"
    print(f"  PASS: scores_kge.shape = {scores_kge.shape}")

    # --- Test 2: forward_sasrec ---
    print("\n--- Test 2: forward_sasrec ---")
    item_seq = torch.randint(1, NUM_ENTITIES, (BATCH, 50))
    item_seq_len = torch.randint(1, 51, (BATCH,))
    user_repr = model.forward_sasrec(item_seq, item_seq_len)
    assert user_repr.shape == (BATCH, KGE_DIM), f"Expected ({BATCH}, {KGE_DIM}), got {user_repr.shape}"
    print(f"  PASS: user_repr.shape = {user_repr.shape} (400 dimensioni senza projection)")

    # --- Test 3: score_items_sasrec ---
    print("\n--- Test 3: score_items_sasrec ---")
    candidates = torch.arange(NUM_ENTITIES)
    scores_rec = model.score_items_sasrec(user_repr.detach(), candidates)
    assert scores_rec.shape == (BATCH, NUM_ENTITIES)
    print(f"  PASS: scores_rec.shape = {scores_rec.shape}")

    # --- Test 4: forward() esplicito lancia errore ---
    print("\n--- Test 4: forward() lancia NotImplementedError ---")
    try:
        model(h, r, t)
        print("  FAIL: doveva lanciare NotImplementedError")
    except NotImplementedError:
        print("  PASS: NotImplementedError lanciato correttamente")

    # --- Test 5: SharedEmbedding non duplicato ---
    print("\n--- Test 5: parametri non duplicati ---")
    total_params = sum(p.numel() for p in model.parameters())
    shared_params = NUM_ENTITIES * KGE_DIM
    assert total_params < shared_params * 3, f"Probabile duplicazione: {total_params:,} parametri"
    print(f"  PASS: total params = {total_params:,} (nessuna duplicazione evidente)")

    # --- Test 6: gradienti fluiscono allo SharedEmbedding da entrambi i task ---
    print("\n--- Test 6: gradient flow da Task A e Task B verso SharedEmbedding ---")
    model.shared_embedding.embedding.weight.grad = None
    scores_kge = model.forward_kge(h, r, t)
    scores_kge.sum().backward()
    assert model.shared_embedding.embedding.weight.grad is not None
    grad_a = model.shared_embedding.embedding.weight.grad.abs().sum().item()

    model.shared_embedding.embedding.weight.grad = None
    user_repr = model.forward_sasrec(item_seq, item_seq_len)
    candidates = torch.arange(NUM_ENTITIES)
    scores = model.score_items_sasrec(user_repr, candidates)
    scores.sum().backward()
    grad_b = model.shared_embedding.embedding.weight.grad.abs().sum().item()

    assert grad_a > 0 and grad_b > 0
    print(f"  PASS: grad da Task A = {grad_a:.4f}, grad da Task B = {grad_b:.4f}")

    print("\n" + "=" * 70)
    print("JOINT ALTERNATE MODEL SMOKE TEST PASSED")
    print("=" * 70)
"""
TaskBSASRec: the SASRec model for the sequential recommendation task,
operating on top of SharedEmbedding via a projection layer.
----------------------------------------------------------------------------
Where to put this file:
    B2B_alternate_learning/models/task_b_sasrec.py
----------------------------------------------------------------------------
"""

import os
import sys
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
# Building blocks: replicate RecBole's SASRec components exactly
# =========================================================================== #

class MultiHeadAttention(nn.Module):
    """RecBole-style multi-head self-attention block."""
    def __init__(self, n_heads, hidden_size, attn_dropout, hidden_dropout, layer_norm_eps):
        super().__init__()
        assert hidden_size % n_heads == 0
        self.n_heads = n_heads
        self.head_size = hidden_size // n_heads
        self.all_head_size = n_heads * self.head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout)

    def transpose_for_scores(self, x):
        # (B, T, H) -> (B, n_heads, T, head_size)
        new_shape = x.size()[:-1] + (self.n_heads, self.head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        Q = self.transpose_for_scores(self.query(hidden_states))
        K = self.transpose_for_scores(self.key(hidden_states))
        V = self.transpose_for_scores(self.value(hidden_states))

        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_size)
        scores = scores + attention_mask
        probs = self.softmax(scores)
        probs = self.attn_dropout(probs)
        ctx = torch.matmul(probs, V)
        ctx = ctx.permute(0, 2, 1, 3).contiguous()
        new_shape = ctx.size()[:-2] + (self.all_head_size,)
        ctx = ctx.view(*new_shape)

        out = self.dense(ctx)
        out = self.out_dropout(out)
        out = self.LayerNorm(out + hidden_states)
        return out


class FeedForward(nn.Module):
    """RecBole-style position-wise feed-forward block."""
    def __init__(self, hidden_size, inner_size, hidden_dropout, layer_norm_eps, hidden_act='gelu'):
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout)
        self.act = nn.GELU() if hidden_act == 'gelu' else nn.ReLU()

    def forward(self, x):
        h = self.dense_1(x)
        h = self.act(h)
        h = self.dense_2(h)
        h = self.dropout(h)
        return self.LayerNorm(h + x)


class TransformerLayer(nn.Module):
    """One block: multi-head attention + feed-forward."""
    def __init__(self, n_heads, hidden_size, inner_size, attn_dropout,
                 hidden_dropout, layer_norm_eps, hidden_act='gelu'):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, attn_dropout, hidden_dropout, layer_norm_eps
        )
        self.feed_forward = FeedForward(
            hidden_size, inner_size, hidden_dropout, layer_norm_eps, hidden_act
        )

    def forward(self, hidden_states, attention_mask):
        h = self.multi_head_attention(hidden_states, attention_mask)
        h = self.feed_forward(h)
        return h


class TransformerEncoder(nn.Module):
    """Stack of n_layers TransformerLayers (mirrors RecBole's TransformerEncoder)."""
    def __init__(self, n_layers, n_heads, hidden_size, inner_size,
                 attn_dropout, hidden_dropout, layer_norm_eps, hidden_act='gelu'):
        super().__init__()
        self.layer = nn.ModuleList([
            TransformerLayer(
                n_heads, hidden_size, inner_size, attn_dropout,
                hidden_dropout, layer_norm_eps, hidden_act,
            )
            for _ in range(n_layers)
        ])

    def forward(self, hidden_states, attention_mask):
        for layer in self.layer:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


# =========================================================================== #
# THE TaskBSASRec CLASS
# =========================================================================== #

class TaskBSASRec(nn.Module):
    """
    SASRec sequential recommender that reads its item embeddings from a
    SharedEmbedding (dim 400).
    """
    def __init__(
        self,
        shared_embedding: SharedEmbedding,
        num_items: int,
        n_layers: int = 4,
        n_heads: int = 2,
        hidden_size: int = 400,
        inner_size: int = 256,
        hidden_dropout: float = 0.4,
        attn_dropout: float = 0.3,
        max_seq_length: int = 50,
        layer_norm_eps: float = 1e-12,
        hidden_act: str = 'gelu',
        padding_idx: int = None
    ):
        super().__init__()

        # Store shared_embedding without auto-registering its params
        object.__setattr__(self, 'shared_embedding', shared_embedding)

        self.shared_dim = shared_embedding.embedding_dim   # 400
        self.hidden_size = hidden_size                     # 400
        self.num_items = num_items
        self.max_seq_length = max_seq_length
        self.padding_idx = padding_idx

        assert self.shared_dim == self.hidden_size, "shared_dim e hidden_size devono coincidere senza projection layer!"

        # ---- Position embedding (private to SASRec) ----
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)

        # ---- Transformer encoder ----
        self.trm_encoder = TransformerEncoder(
            n_layers=n_layers,
            n_heads=n_heads,
            hidden_size=hidden_size,
            inner_size=inner_size,
            attn_dropout=attn_dropout,
            hidden_dropout=hidden_dropout,
            layer_norm_eps=layer_norm_eps,
            hidden_act=hidden_act,
        )

        # ---- Final layer norm + dropout ----
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout)

        logger.info(
            f"Initialized TaskBSASRec: "
            f"direct 400-dim (NO projection), "
            f"{n_layers} transformer layers x {n_heads} heads, "
            f"max_seq_len={max_seq_length}"
        )

    def get_attention_mask(self, item_seq):
        """Causal attention mask for left-to-right sequential modeling."""

        # The attention mask must not look for > 0 but must look for unified padding.
        if self.padding_idx is not None:
            attention_mask = (item_seq != self.padding_idx).long()
        else:
            attention_mask = (item_seq > 0).long()

        ext_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
        max_len = item_seq.size(-1)
        attn_shape = (1, max_len, max_len)
        subseq_mask = torch.triu(torch.ones(attn_shape), diagonal=1).long().to(item_seq.device)
        subseq_mask = subseq_mask == 0  # (1, T, T)
        ext_mask = ext_mask * subseq_mask.unsqueeze(1)  # (B, 1, T, T)
        ext_mask = (1.0 - ext_mask.float()) * -10000.0
        return ext_mask

    def forward(self, item_seq, item_seq_len):
        # ---- Look up item embeddings from SharedEmbedding (400) ----
        item_emb = self.shared_embedding(item_seq)  # (B, T, 400)

        # ---- Add position embedding ----
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_emb = self.position_embedding(position_ids)  # (B, T, 400)

        # ---- Somma diretta (no projection) ----
        h = item_emb + position_emb
        h = self.LayerNorm(h)
        h = self.dropout(h)

        # ---- Apply transformer encoder ----
        attention_mask = self.get_attention_mask(item_seq)
        h = self.trm_encoder(h, attention_mask)  # (B, T, 400)

        # ---- Pick the representation at the last valid position ----
        gather_idx = (item_seq_len - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, h.size(-1))
        user_repr = h.gather(dim=1, index=gather_idx).squeeze(1)  # (B, 400)

        return user_repr

    def score_all_items(self, user_repr, candidate_unified_ids):
        # ---- Match diretto tra 400 (user) e 400 (item) ----
        cand_emb_400 = self.shared_embedding(candidate_unified_ids)  # (N, 400)
        return torch.matmul(user_repr, cand_emb_400.t())

    def load_self_attention_from_recbole(self, recbole_state_dict, verbose=True):
        if verbose:
            logger.info("\n--- Warm start: loading SASRec self-attention from RecBole ---")
            logger.info("    Expect to load 67 tensors (68 total - 1 item_embedding)")

        own_state = self.state_dict()
        loaded = 0
        skipped_item = 0
        not_found = []

        for key, val in recbole_state_dict.items():
            if key.startswith('item_embedding.'):
                skipped_item += 1
                continue
            if key in own_state:
                if own_state[key].shape == val.shape:
                    own_state[key].copy_(val)
                    loaded += 1
                else:
                    not_found.append((key, tuple(val.shape), tuple(own_state[key].shape)))
            else:
                not_found.append((key, tuple(val.shape), 'KEY MISSING'))

        if verbose:
            logger.info(f"  Loaded {loaded} parameter tensors from RecBole")
            logger.info(f"  Skipped item_embedding (intentional): {skipped_item}")
            if not_found:
                logger.warning(f"  Could not load {len(not_found)} keys:")
                for k, src, dst in not_found[:5]:
                    logger.warning(f"    {k}: src={src} dst={dst}")


    # ----------------------------------------------------------------------- #
    # Warm-start from a RecBole SASRec .pth checkpoint
    # ----------------------------------------------------------------------- #

    def load_self_attention_from_recbole(self, recbole_state_dict, verbose=True):
        """
        Load RecBole SASRec's transformer + position embedding weights into us.

        Note: this does NOT load the original 64-dim item_embedding from RecBole,
        because in the new architecture items come from SharedEmbedding -> projection.
        """
        if verbose:
            logger.info("\n--- Warm start: loading SASRec self-attention from RecBole ---")

        # RecBole saves keys like:
        #   item_embedding.weight                (we DO NOT load this)
        #   position_embedding.weight            -> self.position_embedding.weight
        #   trm_encoder.layer.{i}.{module}.{...} -> self.trm_encoder.layer.{i}.{...}
        #   LayerNorm.weight, LayerNorm.bias     -> self.LayerNorm.{}
        own_state = self.state_dict()
        loaded = 0
        skipped_item = 0
        not_found = []

        for key, val in recbole_state_dict.items():
            if key.startswith('item_embedding.'):
                skipped_item += 1
                continue
            if key in own_state:
                if own_state[key].shape == val.shape:
                    own_state[key].copy_(val)
                    loaded += 1
                else:
                    not_found.append((key, tuple(val.shape), tuple(own_state[key].shape)))
            else:
                not_found.append((key, tuple(val.shape), 'KEY MISSING'))

        if verbose:
            logger.info(f"  Loaded {loaded} parameter tensors from RecBole")
            logger.info(f"  Skipped item_embedding (intentional): {skipped_item}")
            if not_found:
                logger.warning(f"  Could not load {len(not_found)} keys:")
                for k, src, dst in not_found[:5]:
                    logger.warning(f"    {k}: src={src} dst={dst}")


# =========================================================================== #
# THE TaskBSASRecLegacy CLASS (Step 3 replica, used ONLY for sanity-check)
# =========================================================================== #

class TaskBSASRecLegacy(nn.Module):
    """
    Faithful reproduction of RecBole's SASRec, with its OWN 64-dim item embedding.

    This is used ONLY to verify that we can reproduce R@20 = 0.369 from Step 3
    inside our infrastructure. It does NOT use SharedEmbedding. It is a sanity
    check, not a model we will use for alternate learning.
    """
    def __init__(
        self,
        num_items: int,
        n_layers: int = 4,
        n_heads: int = 2,
        hidden_size: int = 400,
        inner_size: int = 256,
        hidden_dropout: float = 0.4,
        attn_dropout: float = 0.3,
        max_seq_length: int = 50,
        layer_norm_eps: float = 1e-12,
        hidden_act: str = 'gelu',
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.num_items = num_items

        # OWN item embedding (64-dim, like Step 3)
        self.item_embedding = nn.Embedding(num_items, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=n_layers, n_heads=n_heads, hidden_size=hidden_size,
            inner_size=inner_size, attn_dropout=attn_dropout,
            hidden_dropout=hidden_dropout, layer_norm_eps=layer_norm_eps,
            hidden_act=hidden_act,
        )
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout)

        logger.info(f"Initialized TaskBSASRecLegacy: own 64-dim item embedding (Step 3 replica)")

    def get_attention_mask(self, item_seq):
        attention_mask = (item_seq > 0).long()
        ext_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        max_len = item_seq.size(-1)
        attn_shape = (1, max_len, max_len)
        subseq_mask = torch.triu(torch.ones(attn_shape), diagonal=1).long().to(item_seq.device)
        subseq_mask = subseq_mask == 0
        ext_mask = ext_mask * subseq_mask.unsqueeze(1)
        ext_mask = (1.0 - ext_mask.float()) * -10000.0
        return ext_mask

    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_emb = self.position_embedding(position_ids)
        h = item_emb + position_emb
        h = self.LayerNorm(h)
        h = self.dropout(h)
        attn_mask = self.get_attention_mask(item_seq)
        h = self.trm_encoder(h, attn_mask)
        gather_idx = (item_seq_len - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, h.size(-1))
        user_repr = h.gather(dim=1, index=gather_idx).squeeze(1)
        return user_repr

    def score_all_items(self, user_repr):
        # Score against ALL items in the legacy embedding
        return torch.matmul(user_repr, self.item_embedding.weight.t())

    def load_full_recbole(self, recbole_state_dict, verbose=True):
        """Load every key from RecBole's state_dict, including item_embedding."""
        if verbose:
            logger.info("\n--- Loading FULL RecBole SASRec (legacy mode) ---")
        own_state = self.state_dict()
        loaded, missing = 0, []
        for k, v in recbole_state_dict.items():
            if k in own_state and own_state[k].shape == v.shape:
                own_state[k].copy_(v)
                loaded += 1
            else:
                missing.append(k)
        if verbose:
            logger.info(f"  Loaded {loaded} tensors, missing/mismatched: {len(missing)}")


# =========================================================================== #
# SMOKE TEST
# =========================================================================== #
if __name__ == "__main__":

    print("=" * 70)
    print("TASK B (SASRec) SMOKE TEST")
    print("=" * 70)

    NUM_ENTITIES = 24_957
    NUM_ITEMS_RECBOLE = 21_135  # incl. padding at 0
    SHARED_DIM = 400
    SASREC_DIM = 400
    BATCH = 4
    SEQ_LEN = 50

    # Build SharedEmbedding (400-dim)
    shared = SharedEmbedding(num_entities=NUM_ENTITIES, embedding_dim=SHARED_DIM, padding_idx=0)

    # Build TaskBSASRec on top
    task_b = TaskBSASRec(
        shared_embedding=shared,
        num_items=NUM_ITEMS_RECBOLE,
        n_layers=4, n_heads=2, hidden_size=SASREC_DIM, inner_size=256,
        hidden_dropout=0.4, attn_dropout=0.3, max_seq_length=SEQ_LEN,
    )

    # ---- Test 1: forward pass ----
    print("\n--- Test 1: forward pass ---")
    item_seq = torch.randint(1, NUM_ENTITIES, (BATCH, SEQ_LEN))
    item_seq_len = torch.randint(1, SEQ_LEN + 1, (BATCH,))
    user_repr = task_b(item_seq, item_seq_len)
    assert user_repr.shape == (BATCH, SASREC_DIM)
    print(f"  PASS: user_repr shape = {user_repr.shape}")

    # ---- Test 2: gradient flow ----
    print("\n--- Test 2: gradient flow ---")
    shared.embedding.weight.grad = None
    user_repr = task_b(item_seq, item_seq_len)
    user_repr.sum().backward()
    assert shared.embedding.weight.grad is not None
    grad_abs_sum = shared.embedding.weight.grad[1:].abs().sum().item() 
    assert grad_abs_sum > 0, "No gradient flowed to SharedEmbedding!"
    # Print modificato per l'assenza della proiezione
    print(f"  PASS: gradients flow to shared, transformer")

    # ---- Test 3: score_all_items ----
    print("\n--- Test 3: score_all_items ---")
    candidates = torch.arange(NUM_ENTITIES)
    scores = task_b.score_all_items(user_repr.detach(), candidates)
    assert scores.shape == (BATCH, NUM_ENTITIES)
    print(f"  PASS: scores shape = {scores.shape}")

    # ---- Test 4: shared_embedding NOT registered twice ----
    print("\n--- Test 4: shared_embedding NOT registered as child module ---")
    own_params = sum(p.numel() for p in task_b.parameters())
    shared_params = NUM_ENTITIES * SHARED_DIM
    assert own_params < shared_params, f"task_b owns {own_params:,} params, expected < {shared_params:,}"
    print(f"  PASS: task_b owns {own_params:,} params (< shared's {shared_params:,})")

    # ---- Test 5: TaskBSASRecLegacy works ----
    print("\n--- Test 5: TaskBSASRecLegacy basic forward ---")
    legacy = TaskBSASRecLegacy(num_items=NUM_ITEMS_RECBOLE)
    item_seq_legacy = torch.randint(1, NUM_ITEMS_RECBOLE, (BATCH, SEQ_LEN))
    user_repr_legacy = legacy(item_seq_legacy, item_seq_len)
    assert user_repr_legacy.shape == (BATCH, SASREC_DIM)
    legacy_scores = legacy.score_all_items(user_repr_legacy)
    assert legacy_scores.shape == (BATCH, NUM_ITEMS_RECBOLE)
    print(f"  PASS: TaskBSASRecLegacy forward + score work")

    # ---- Test 6: load_self_attention_from_recbole with fake state_dict ----
    print("\n--- Test 6: warm start of self-attention layers ---")
    fake_recbole_sd = {
        'item_embedding.weight': torch.randn(NUM_ITEMS_RECBOLE, SASREC_DIM),  # IGNORED
        'position_embedding.weight': torch.randn(SEQ_LEN, SASREC_DIM),
        'LayerNorm.weight': torch.ones(SASREC_DIM),
        'LayerNorm.bias': torch.zeros(SASREC_DIM),
    }

    # Genera iterativamente tutti i pesi (attention + feed forward) per n_layers = 4
    for i in range(4):
        for sub in ['query', 'key', 'value', 'dense']:
            fake_recbole_sd[f'trm_encoder.layer.{i}.multi_head_attention.{sub}.weight'] = torch.randn(SASREC_DIM, SASREC_DIM)
            fake_recbole_sd[f'trm_encoder.layer.{i}.multi_head_attention.{sub}.bias'] = torch.randn(SASREC_DIM)
        fake_recbole_sd[f'trm_encoder.layer.{i}.multi_head_attention.LayerNorm.weight'] = torch.randn(SASREC_DIM)
        fake_recbole_sd[f'trm_encoder.layer.{i}.multi_head_attention.LayerNorm.bias'] = torch.randn(SASREC_DIM)
        for sub, shape_w in [('dense_1', (256, SASREC_DIM)), ('dense_2', (SASREC_DIM, 256))]:
            fake_recbole_sd[f'trm_encoder.layer.{i}.feed_forward.{sub}.weight'] = torch.randn(*shape_w)
            fake_recbole_sd[f'trm_encoder.layer.{i}.feed_forward.{sub}.bias'] = torch.randn(shape_w[0])
        fake_recbole_sd[f'trm_encoder.layer.{i}.feed_forward.LayerNorm.weight'] = torch.randn(SASREC_DIM)
        fake_recbole_sd[f'trm_encoder.layer.{i}.feed_forward.LayerNorm.bias'] = torch.randn(SASREC_DIM)

    task_b.load_self_attention_from_recbole(fake_recbole_sd, verbose=True)
    print(f"  PASS: load_self_attention_from_recbole runs without errors")

    print("\n" + "=" * 70)
    print("TASK B (SASRec) SMOKE TEST PASSED")
    print("=" * 70)
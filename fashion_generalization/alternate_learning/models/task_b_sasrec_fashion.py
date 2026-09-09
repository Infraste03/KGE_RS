"""
task_b_sasrec_fashion.py
=========================

TaskBSASRec: SASRec model for Task B of Fashion alternate learning.

It operates ON TOP of SharedEmbedding (182,984 x 64):
  - SharedEmbedding provides the item embeddings (shared with Task A)
  - TaskBSASRec owns the position embedding, Transformer encoder,
    LayerNorm, and dropout layers (all private to Task B)

Differences compared with B2B:
  - hidden_size = 64 (not 400)
  - n_layers = 4, n_heads = 4 (best Fashion HPO trial 17)
  - inner_size = 256 (= 4 * hidden_size, RecBole default)
  - padding_idx = 182983 (not 0)
    In Fashion, RecBole uses id=0 as PAD, but in the unified space
    id=0 is already occupied by the first brand. Therefore, unified
    padding is the last row of SharedEmbedding (182983).
    The attention mask uses item_seq != 182983.
  - max_seq_length = 50

File location:
    fashion_generalization/alternate_learning/models/task_b_sasrec_fashion.py
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
from shared_embedding_fashion import SharedEmbedding

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =========================================================================== #
# Building blocks — exact replica of RecBole SASRec
# (identical to B2B, no architectural differences)
# =========================================================================== #

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention exactly as implemented in RecBole."""
    def __init__(self, n_heads, hidden_size, attn_dropout, hidden_dropout, layer_norm_eps):
        super().__init__()
        assert hidden_size % n_heads == 0, \
            f"hidden_size={hidden_size} is not divisible by n_heads={n_heads}"
        self.n_heads   = n_heads
        self.head_size = hidden_size // n_heads
        self.all_head_size = n_heads * self.head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key   = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.softmax      = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dense        = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm    = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout  = nn.Dropout(hidden_dropout)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.n_heads, self.head_size)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        Q = self.transpose_for_scores(self.query(hidden_states))
        K = self.transpose_for_scores(self.key(hidden_states))
        V = self.transpose_for_scores(self.value(hidden_states))

        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_size)
        scores = scores + attention_mask
        probs  = self.attn_dropout(self.softmax(scores))
        ctx    = torch.matmul(probs, V)
        ctx    = ctx.permute(0, 2, 1, 3).contiguous()
        ctx    = ctx.view(ctx.size()[:-2] + (self.all_head_size,))

        out = self.out_dropout(self.dense(ctx))
        return self.LayerNorm(out + hidden_states)


class FeedForward(nn.Module):
    """Position-wise feed-forward exactly as implemented in RecBole."""
    def __init__(self, hidden_size, inner_size, hidden_dropout, layer_norm_eps,
                 hidden_act='gelu'):
        super().__init__()
        self.dense_1  = nn.Linear(hidden_size, inner_size)
        self.dense_2  = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout  = nn.Dropout(hidden_dropout)
        self.act      = nn.GELU() if hidden_act == 'gelu' else nn.ReLU()

    def forward(self, x):
        h = self.dense_2(self.act(self.dense_1(x)))
        return self.LayerNorm(self.dropout(h) + x)


class TransformerLayer(nn.Module):
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
        return self.feed_forward(
            self.multi_head_attention(hidden_states, attention_mask)
        )


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, n_heads, hidden_size, inner_size,
                 attn_dropout, hidden_dropout, layer_norm_eps, hidden_act='gelu'):
        super().__init__()
        self.layer = nn.ModuleList([
            TransformerLayer(n_heads, hidden_size, inner_size, attn_dropout,
                             hidden_dropout, layer_norm_eps, hidden_act)
            for _ in range(n_layers)
        ])

    def forward(self, hidden_states, attention_mask):
        for layer in self.layer:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


# =========================================================================== #
# TaskBSASRec — uses SharedEmbedding (64-dim)
# =========================================================================== #

class TaskBSASRec(nn.Module):
    """
    SASRec that reads item embeddings from SharedEmbedding (64-dim).

    Parameters
    ----------
    shared_embedding : SharedEmbedding
        Shared matrix (182,984 x 64). padding_idx=182983.
    n_layers : int
        Number of Transformer layers. For Fashion (trial 17): 4.
    n_heads : int
        Number of attention heads. For Fashion (trial 17): 4.
    hidden_size : int
        Embedding dimension. Must match shared_embedding.embedding_dim.
        For Fashion: 64.
    inner_size : int
        Internal feed-forward dimension. For Fashion: 256
        (= 4 * 64, RecBole default).
    hidden_dropout : float
        Dropout applied to hidden layers. For Fashion (trial 17): 0.5.
    attn_dropout : float
        Attention dropout. For Fashion (trial 17): 0.3.
    max_seq_length : int
        Maximum sequence length. For Fashion: 50.
    unified_padding_idx : int
        Padding ID in the unified space. For Fashion: 182983.
        Used by the attention mask to identify padding positions.
        DIFFERENT from B2B, where it was 0.
    """

    def __init__(
        self,
        shared_embedding: SharedEmbedding,
        n_layers: int         = 4,
        n_heads: int          = 4,
        hidden_size: int      = 64,
        inner_size: int       = 256,
        hidden_dropout: float = 0.5,
        attn_dropout: float   = 0.3,
        max_seq_length: int   = 50,
        layer_norm_eps: float = 1e-12,
        hidden_act: str       = 'gelu',
        unified_padding_idx: int = 182_983,
    ):
        super().__init__()

        if shared_embedding.embedding_dim != hidden_size:
            raise ValueError(
                f"shared_embedding.embedding_dim={shared_embedding.embedding_dim} "
                f"!= hidden_size={hidden_size}. "
                f"They must match: there is no projection layer in the Fashion architecture."
            )

        # Do NOT register shared_embedding as a child module —
        # otherwise its parameters would appear twice in the optimizer
        object.__setattr__(self, 'shared_embedding', shared_embedding)

        self.hidden_size        = hidden_size
        self.max_seq_length     = max_seq_length
        self.unified_padding_idx = unified_padding_idx

        # Position embedding (private to Task B)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)

        # Transformer encoder
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

        # Final LayerNorm and dropout
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout   = nn.Dropout(hidden_dropout)

        logger.info(
            f"TaskBSASRec Fashion initialized: "
            f"{n_layers} layers x {n_heads} heads, "
            f"hidden={hidden_size}, inner={inner_size}, "
            f"max_seq={max_seq_length}, "
            f"unified_padding_idx={unified_padding_idx}"
        )

    def get_attention_mask(self, item_seq: torch.Tensor) -> torch.Tensor:
        """
        Causal attention mask for left-to-right sequential modeling.

        CRITICAL DIFFERENCE compared with B2B:
        In B2B, unified padding was 0, therefore (item_seq > 0) was used.
        In Fashion, unified padding is 182983, therefore we use
        (item_seq != unified_padding_idx).

        This is essential: if (item_seq > 0) were used here, brand_*
        entities with IDs 0..17510 would be incorrectly treated as padding.
        """
        attention_mask = (item_seq != self.unified_padding_idx).long()  # (B, T)
        ext_mask = attention_mask.unsqueeze(1).unsqueeze(2)              # (B, 1, 1, T)

        max_len = item_seq.size(-1)
        subseq_mask = torch.triu(
            torch.ones((1, max_len, max_len), dtype=torch.long),
            diagonal=1
        ).to(item_seq.device)
        subseq_mask = (subseq_mask == 0)  # (1, T, T)

        ext_mask = ext_mask * subseq_mask.unsqueeze(1)        # (B, 1, T, T)
        ext_mask = (1.0 - ext_mask.float()) * -10000.0
        return ext_mask

    def forward(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        item_seq : LongTensor (B, T)
            Sequence of unified item IDs. Padding positions contain
            unified_padding_idx (182983), NOT 0.
        item_seq_len : LongTensor (B,)
            Actual length of each sequence (excluding padding).

        Returns
        -------
        user_repr : FloatTensor (B, 64)
        """
        # Look up item embeddings from SharedEmbedding
        item_emb = self.shared_embedding(item_seq)  # (B, T, 64)

        # Position embedding
        position_ids  = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        ).unsqueeze(0).expand_as(item_seq)
        position_emb  = self.position_embedding(position_ids)  # (B, T, 64)

        # Direct sum (no projection, dimensions already match)
        h = self.LayerNorm(item_emb + position_emb)
        h = self.dropout(h)

        # Transformer encoder with causal mask
        h = self.trm_encoder(h, self.get_attention_mask(item_seq))  # (B, T, 64)

        # Extract the representation at the last valid position
        gather_idx = (item_seq_len - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, h.size(-1))
        user_repr  = h.gather(dim=1, index=gather_idx).squeeze(1)   # (B, 64)
        return user_repr

    def score_all_items(
        self,
        user_repr: torch.Tensor,
        candidate_unified_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dot product between user_repr and candidate embeddings.

        Parameters
        ----------
        user_repr : FloatTensor (B, 64)
        candidate_unified_ids : LongTensor (N,)
            Unified IDs of candidate items. Typically
            torch.LongTensor(space.item_unified_ids) — items only,
            NOT all 182,984 IDs (which also include brands and categories).

        Returns
        -------
        scores : FloatTensor (B, N)
        """
        cand_emb = self.shared_embedding(candidate_unified_ids)  # (N, 64)
        return torch.matmul(user_repr, cand_emb.t())             # (B, N)

    def load_self_attention_from_recbole(
        self,
        recbole_state_dict: dict,
        verbose: bool = True,
    ):
        """
        Load Transformer + position embedding weights from a RecBole checkpoint.

        item_embedding.weight is NOT loaded because Fashion item embeddings
        live in SharedEmbedding (already initialized from TransE).

        Fashion trial 17: n_layers=4, n_heads=4, hidden_size=64, inner_size=256.
        Expected tensors from RecBole:
          - position_embedding.weight         : (50, 64)
          - LayerNorm.weight, LayerNorm.bias  : (64,)
          - trm_encoder.layer.{0..3}.multi_head_attention.{query,key,value,dense}.{weight,bias}
          - trm_encoder.layer.{0..3}.multi_head_attention.LayerNorm.{weight,bias}
          - trm_encoder.layer.{0..3}.feed_forward.{dense_1,dense_2}.{weight,bias}
          - trm_encoder.layer.{0..3}.feed_forward.LayerNorm.{weight,bias}
        Expected total: 3 + 4*(8+2+4+2) = 3 + 4*16 = 3 + 64 = 67 tensors
        (68 total tensors in the RecBole checkpoint - 1 item_embedding.weight)
        """
        if verbose:
            logger.info("\n--- Warm start: loading SASRec from RecBole (Fashion) ---")
            logger.info("    item_embedding.weight is SKIPPED (it lives in SharedEmbedding)")
            logger.info("    Expected: 67 tensors (68 total - 1 item_embedding)")

        own_state    = self.state_dict()
        loaded       = 0
        skipped_item = 0
        not_found    = []

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

        self.load_state_dict(own_state)

        if verbose:
            logger.info(f"  Tensors loaded                   : {loaded}")
            logger.info(f"  item_embedding skipped           : {skipped_item}")
            if not_found:
                logger.warning(f"  NOT loaded ({len(not_found)}):")
                for k, src, dst in not_found[:10]:
                    logger.warning(f"    {k}: src={src} dst={dst}")
            else:
                logger.info(f"  No missing tensors: warm start complete")


# =========================================================================== #
# TaskBSASRecLegacy — exact replica of RecBole SASRec with its own embedding
# Used ONLY to verify that our code reproduces RecBole metrics
# =========================================================================== #

class TaskBSASRecLegacy(nn.Module):
    """
    Faithful reproduction of RecBole SASRec with its own item_embedding (64-dim).

    It does NOT use SharedEmbedding. It is used ONLY for the sanity check:
    it verifies that our infrastructure reproduces R@20 ~ 0.0951
    (trial 17, test_from_best_valid_recall) using checkpoint weights.

    It is never used during alternate-learning training.
    """

    def __init__(
        self,
        num_items: int,           # 165,470 (165,469 real items + 1 PAD)
        n_layers: int   = 4,
        n_heads: int    = 4,
        hidden_size: int = 64,
        inner_size: int  = 256,
        hidden_dropout: float = 0.5,
        attn_dropout: float   = 0.3,
        max_seq_length: int   = 50,
        layer_norm_eps: float = 1e-12,
        hidden_act: str       = 'gelu',
    ):
        super().__init__()
        self.hidden_size    = hidden_size
        self.max_seq_length = max_seq_length
        self.num_items      = num_items

        # Own item_embedding with padding_idx=0 (as in RecBole)
        self.item_embedding  = nn.Embedding(num_items, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=n_layers, n_heads=n_heads, hidden_size=hidden_size,
            inner_size=inner_size, attn_dropout=attn_dropout,
            hidden_dropout=hidden_dropout, layer_norm_eps=layer_norm_eps,
            hidden_act=hidden_act,
        )
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout   = nn.Dropout(hidden_dropout)

        logger.info(
            f"TaskBSASRecLegacy Fashion: own item_embedding "
            f"({num_items} x {hidden_size}), "
            f"{n_layers} layers x {n_heads} heads"
        )

    def get_attention_mask(self, item_seq: torch.Tensor) -> torch.Tensor:
        """
        In Legacy mode, item_seq > 0 is used because IDs are raw RecBole IDs
        (padding=0 according to the standard RecBole convention).
        """
        attention_mask = (item_seq > 0).long()
        ext_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        max_len  = item_seq.size(-1)
        subseq_mask = torch.triu(
            torch.ones((1, max_len, max_len), dtype=torch.long), diagonal=1
        ).to(item_seq.device)
        subseq_mask = (subseq_mask == 0)
        ext_mask = ext_mask * subseq_mask.unsqueeze(1)
        ext_mask = (1.0 - ext_mask.float()) * -10000.0
        return ext_mask

    def forward(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> torch.Tensor:
        item_emb    = self.item_embedding(item_seq)
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        ).unsqueeze(0).expand_as(item_seq)
        position_emb = self.position_embedding(position_ids)
        h = self.LayerNorm(item_emb + position_emb)
        h = self.dropout(h)
        h = self.trm_encoder(h, self.get_attention_mask(item_seq))
        gather_idx = (item_seq_len - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, h.size(-1))
        return h.gather(dim=1, index=gather_idx).squeeze(1)

    def score_all_items(self, user_repr: torch.Tensor) -> torch.Tensor:
        """Score against ALL items (raw RecBole space)."""
        return torch.matmul(user_repr, self.item_embedding.weight.t())

    def load_full_recbole(self, recbole_state_dict: dict, verbose: bool = True):
        """Load ALL weights from the RecBole checkpoint (including item_embedding)."""
        if verbose:
            logger.info("\n--- FULL RecBole SASRec loading (legacy mode) ---")
        own_state = self.state_dict()
        loaded, not_found = 0, []
        for k, v in recbole_state_dict.items():
            if k in own_state and own_state[k].shape == v.shape:
                own_state[k].copy_(v)
                loaded += 1
            else:
                not_found.append((k, tuple(v.shape)))
        self.load_state_dict(own_state)
        if verbose:
            logger.info(f"  Tensors loaded     : {loaded}")
            if not_found:
                logger.warning(f"  Not loaded         : {len(not_found)}")
                for k, s in not_found[:5]:
                    logger.warning(f"    {k}: {s}")


# =========================================================================== #
# STANDALONE SMOKE TEST
#
# Real Fashion dimensions:
#   - SharedEmbedding: 182,984 x 64, padding_idx=182983
#   - unified item IDs: 17514..182982 (165,469 items)
#   - n_layers=4, n_heads=4, hidden_size=64, inner_size=256
#   - max_seq_length=50
# =========================================================================== #

if __name__ == "__main__":

    print("=" * 70)
    print("TASK B SASREC FASHION — SMOKE TEST")
    print("=" * 70)

    NUM_KG_ENTITIES  = 182_983
    NUM_ROWS         = 182_984   # + 1 padding
    PADDING_IDX      = 182_983
    NUM_ITEMS_REAL   = 165_469
    ITEM_ID_START    = 17_514    # first item in the unified space
    NUM_ITEMS_RECBOLE = 165_470  # real items + 1 RecBole PAD
    HIDDEN           = 64
    BATCH            = 4
    SEQ_LEN          = 50

    shared = SharedEmbedding(
        num_entities=NUM_ROWS,
        embedding_dim=HIDDEN,
        padding_idx=PADDING_IDX,
    )

    task_b = TaskBSASRec(
        shared_embedding=shared,
        n_layers=4,
        n_heads=4,
        hidden_size=HIDDEN,
        inner_size=256,
        hidden_dropout=0.5,
        attn_dropout=0.3,
        max_seq_length=SEQ_LEN,
        unified_padding_idx=PADDING_IDX,
    )


    # ---- Test 1: forward pass with real unified IDs ----
    print("\n--- Test 1: forward pass (unified item IDs) ---")
    # Sequences with real unified item IDs (17514..182982) + padding (182983)
    item_seq = torch.randint(ITEM_ID_START, NUM_KG_ENTITIES, (BATCH, SEQ_LEN))
    # Add some trailing padding to some sequences
    item_seq_len = torch.randint(3, SEQ_LEN, (BATCH,))
    for b in range(BATCH):
        item_seq[b, item_seq_len[b]:] = PADDING_IDX

    user_repr = task_b(item_seq, item_seq_len)
    assert user_repr.shape == (BATCH, HIDDEN), f"Incorrect shape: {user_repr.shape}"
    print(f"  PASS: user_repr shape = {user_repr.shape}")

    # ---- Test 2: attention mask with Fashion padding_idx ----
    print("\n--- Test 2: attention mask — padding_idx=182983, not 0 ---")
    mask = task_b.get_attention_mask(item_seq)
    assert mask.shape == (BATCH, 1, SEQ_LEN, SEQ_LEN), \
        f"Incorrect mask shape: {mask.shape}"

    # Padding positions must contain -10000 in the mask
    for b in range(BATCH):
        pad_start = item_seq_len[b].item()
        if pad_start < SEQ_LEN:
            # The column corresponding to padding must contain -10000
            assert (mask[b, 0, :, pad_start] == -10000.0).all(), \
                f"Batch {b}: padding position {pad_start} is not masked!"
    print(f"  PASS: mask shape {mask.shape}, padding positions correctly masked")

    # ---- Test 3: brand/category are NOT treated as padding ----
    print("\n--- Test 3: brand IDs (0..17510) are NOT treated as padding ---")
    # In B2B, item_seq > 0 would have treated brand_0 (id=0) as padding.
    # Here we verify that id=0 is NOT masked.
    seq_with_brand = torch.full((1, SEQ_LEN), PADDING_IDX, dtype=torch.long)
    seq_with_brand[0, 0] = 0     # brand_0 (first brand)
    seq_with_brand[0, 1] = 100   # another brand
    seq_len_brand = torch.tensor([2])

    mask_brand = task_b.get_attention_mask(seq_with_brand)
    # Columns 0 and 1 must be valid (not -10000) in rows 0 and 1
    # (with causal masking, row 0 sees only column 0, row 1 sees columns 0 and 1)
    assert mask_brand[0, 0, 0, 0] != -10000.0, \
        "brand_0 (id=0) was incorrectly masked as padding!"
    assert mask_brand[0, 0, 1, 0] != -10000.0, \
        "brand_0 in column 0 was incorrectly masked!"
    print(f"  PASS: brand_0 (id=0) is NOT masked — correct Fashion padding_idx")
    

    print("\n--- ISOLATED Test: direct gradient from shared_embedding ---")

    # Completely isolated test: direct forward, without task_b
    shared_test = SharedEmbedding(
        num_entities=NUM_ROWS,
        embedding_dim=HIDDEN,
        padding_idx=PADDING_IDX,
    )
    shared_test.embedding.weight.grad = None

    ids = torch.tensor([ITEM_ID_START, ITEM_ID_START + 1])
    out = shared_test(ids)
    out.sum().backward()

    grad_direct = shared_test.embedding.weight.grad
    print(f"  Direct gradient on shared_test: {grad_direct[ITEM_ID_START].abs().sum():.8f}")
    print(f"  (expected > 0)")

    # Now test task_b with THIS shared_test — not with the main smoke-test shared instance
    task_b_test = TaskBSASRec(
        shared_embedding=shared_test,
        n_layers=4, n_heads=4, hidden_size=HIDDEN, inner_size=256,
        hidden_dropout=0.5, attn_dropout=0.3,
        max_seq_length=SEQ_LEN, unified_padding_idx=PADDING_IDX,
    )
    shared_test.embedding.weight.grad = None

    seq_test = torch.full((1, SEQ_LEN), PADDING_IDX, dtype=torch.long)
    seq_test[0, 0] = ITEM_ID_START
    seq_test[0, 1] = ITEM_ID_START + 1
    len_test = torch.tensor([2])

    out2 = task_b_test(seq_test, len_test)
    out2.sum().backward()

    grad2 = shared_test.embedding.weight.grad
    print(f"  Gradient through task_b on shared_test[{ITEM_ID_START}]: {grad2[ITEM_ID_START].abs().sum():.8f}")
    print(f"  Gradient through task_b on shared_test[{ITEM_ID_START+1}]: {grad2[ITEM_ID_START+1].abs().sum():.8f}")
    nonzero = (grad2.abs().sum(dim=1) > 1e-15).nonzero().squeeze()
    print(f"  Rows with gradient != 0: {nonzero.tolist()}")
    
    
    print("\n--- MINIMAL Test: backward through object.__setattr__ ---")

    import torch
    import torch.nn as nn

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(10, 4)
        def forward(self, x):
            return self.emb(x)

    class Outer(nn.Module):
        def __init__(self, inner):
            super().__init__()
            object.__setattr__(self, 'inner', inner)
            self.linear = nn.Linear(4, 4)
        def forward(self, x):
            return self.linear(self.inner(x))

    inner = Inner()
    outer = Outer(inner)

    inner.emb.weight.grad = None
    ids = torch.tensor([1, 2, 3])
    out = outer(ids)
    out.sum().backward()

    g = inner.emb.weight.grad
    print(f"  gradient on inner.emb (through object.__setattr__): {g[1].abs().sum().item():.6f}")
    print(f"  (if 0.0 -> object.__setattr__ breaks the graph)")
    
    # ---- Test 4: gradient flow to shared_embedding ----
    print("\n--- Test 4: gradient flow to shared_embedding ---")

    # Direct verification: shared_embedding.forward() is part of the computational graph
    # The most reliable test is score_all_items, which uses shared_embedding directly
    # without passing through the Transformer
    shared_4 = SharedEmbedding(
        num_entities=NUM_ROWS,
        embedding_dim=HIDDEN,
        padding_idx=PADDING_IDX,
    )
    task_b_4 = TaskBSASRec(
        shared_embedding=shared_4,
        n_layers=4, n_heads=4,
        hidden_size=HIDDEN, inner_size=256,
        hidden_dropout=0.0,
        attn_dropout=0.0,
        max_seq_length=SEQ_LEN,
        unified_padding_idx=PADDING_IDX,
    )

    # Test 4a: gradient through score_all_items (uses shared_embedding directly)
    shared_4.embedding.weight.grad = None
    fake_user_repr = torch.randn(1, HIDDEN, requires_grad=False)
    candidate_ids  = torch.tensor([ITEM_ID_START, ITEM_ID_START + 1, ITEM_ID_START + 100])
    scores_4a = task_b_4.score_all_items(fake_user_repr, candidate_ids)
    scores_4a.sum().backward()

    grad_4a = shared_4.embedding.weight.grad
    assert grad_4a is not None
    for eid in [ITEM_ID_START, ITEM_ID_START + 1, ITEM_ID_START + 100]:
        g = grad_4a[eid].abs().sum().item()
        assert g > 0, f"score_all_items: item {eid} did not receive a gradient!"
        print(f"  [score_all_items] grad[{eid}]: {g:.8f} ✓")
    assert grad_4a[PADDING_IDX].abs().sum().item() == 0.0
    print(f"  [score_all_items] padding grad: 0.0 ✓")
    print(f"  PASS: shared_embedding is part of the score_all_items graph")

    # Test 4b: gradient through forward() with sequence len=2
    # (len=1 with random weights may produce zero gradient due to numerical
    # cancellation through 4 Transformer layers — this is mathematics, not a bug)
    shared_4.embedding.weight.grad = None
    seq4 = torch.full((1, SEQ_LEN), PADDING_IDX, dtype=torch.long)
    seq4[0, 0] = ITEM_ID_START
    seq4[0, 1] = ITEM_ID_START + 1
    len4 = torch.tensor([2])

    user_repr_4 = task_b_4(seq4, len4)
    user_repr_4.sum().backward()

    grad_4b = shared_4.embedding.weight.grad
    g_total = grad_4b.abs().sum().item()
    assert g_total > 0, \
        f"forward(): total gradient is zero! Check the architecture."
    assert grad_4b[PADDING_IDX].abs().sum().item() == 0.0, \
        "Padding row received gradient!"
    assert grad_4b[:ITEM_ID_START].abs().sum().item() == 0.0, \
        "Brand/category rows received unexpected gradient!"

    nonzero = (grad_4b.abs().sum(dim=1) > 1e-10).nonzero().squeeze(-1)
    print(f"  [forward] total gradient: {g_total:.8f} ✓")
    print(f"  [forward] rows with gradient != 0: {nonzero.tolist()}")
    print(f"  [forward] padding grad: 0.0 ✓")
    print(f"  [forward] brand/category grad: 0.0 ✓")
    print(f"  PASS: gradient flows through forward() to SharedEmbedding")

    

    # ---- Test 5: score_all_items ----
    print("\n--- Test 5: score_all_items with unified item IDs ---")
    with torch.no_grad():
        item_seq_t5 = torch.randint(ITEM_ID_START, NUM_KG_ENTITIES, (BATCH, SEQ_LEN))
        len_t5 = torch.full((BATCH,), SEQ_LEN // 2)
        user_repr_t5 = task_b(item_seq_t5, len_t5)
    candidate_ids = torch.arange(ITEM_ID_START, NUM_KG_ENTITIES, dtype=torch.long)
    scores = task_b.score_all_items(user_repr_t5, candidate_ids)
    assert scores.shape == (BATCH, NUM_ITEMS_REAL), \
        f"Incorrect scores shape: {scores.shape}"
    print(f"  PASS: scores shape = {scores.shape} ({BATCH} users x {NUM_ITEMS_REAL:,} items)")

    # ---- Test 6: shared_embedding is NOT a child module ----
    print("\n--- Test 6: shared_embedding is NOT registered as a child module ---")
    own_params    = sum(p.numel() for p in task_b.parameters())
    shared_params = NUM_ROWS * HIDDEN
    assert own_params < shared_params, \
        f"task_b has {own_params:,} params, expected < {shared_params:,}"
    print(f"  PASS: task_b has {own_params:,} own params (< {shared_params:,} shared params)")

    # ---- Test 7: TaskBSASRecLegacy forward + score ----
    print("\n--- Test 7: TaskBSASRecLegacy forward and score ---")
    legacy = TaskBSASRecLegacy(num_items=NUM_ITEMS_RECBOLE)
    item_seq_legacy = torch.randint(1, NUM_ITEMS_RECBOLE, (BATCH, SEQ_LEN))
    user_repr_legacy = legacy(item_seq_legacy, item_seq_len)
    assert user_repr_legacy.shape == (BATCH, HIDDEN)
    legacy_scores = legacy.score_all_items(user_repr_legacy)
    assert legacy_scores.shape == (BATCH, NUM_ITEMS_RECBOLE)
    print(f"  PASS: Legacy forward {user_repr_legacy.shape}, scores {legacy_scores.shape}")

    # ---- Test 8: warm start with fake Fashion state_dict ----
    print("\n--- Test 8: load_self_attention_from_recbole (Fashion, 4 layers x 4 heads) ---")
    # Build a fake state_dict with the keys produced by RecBole for Fashion
    fake_sd = {
        'item_embedding.weight'  : torch.randn(NUM_ITEMS_RECBOLE, HIDDEN),  # SKIPPED
        'position_embedding.weight': torch.randn(SEQ_LEN, HIDDEN),
        'LayerNorm.weight'         : torch.ones(HIDDEN),
        'LayerNorm.bias'           : torch.zeros(HIDDEN),
    }
    for i in range(4):  # n_layers=4
        for sub in ['query', 'key', 'value', 'dense']:
            fake_sd[f'trm_encoder.layer.{i}.multi_head_attention.{sub}.weight'] = \
                torch.randn(HIDDEN, HIDDEN)
            fake_sd[f'trm_encoder.layer.{i}.multi_head_attention.{sub}.bias'] = \
                torch.randn(HIDDEN)
        fake_sd[f'trm_encoder.layer.{i}.multi_head_attention.LayerNorm.weight'] = \
            torch.ones(HIDDEN)
        fake_sd[f'trm_encoder.layer.{i}.multi_head_attention.LayerNorm.bias'] = \
            torch.zeros(HIDDEN)
        # inner_size=256: dense_1: (256, 64), dense_2: (64, 256)
        fake_sd[f'trm_encoder.layer.{i}.feed_forward.dense_1.weight'] = \
            torch.randn(256, HIDDEN)
        fake_sd[f'trm_encoder.layer.{i}.feed_forward.dense_1.bias'] = \
            torch.randn(256)
        fake_sd[f'trm_encoder.layer.{i}.feed_forward.dense_2.weight'] = \
            torch.randn(HIDDEN, 256)
        fake_sd[f'trm_encoder.layer.{i}.feed_forward.dense_2.bias'] = \
            torch.randn(HIDDEN)
        fake_sd[f'trm_encoder.layer.{i}.feed_forward.LayerNorm.weight'] = \
            torch.ones(HIDDEN)
        fake_sd[f'trm_encoder.layer.{i}.feed_forward.LayerNorm.bias'] = \
            torch.zeros(HIDDEN)

    # Verify that position_embedding changes after warm start
    pos_before = task_b.position_embedding.weight.clone()
    task_b.load_self_attention_from_recbole(fake_sd, verbose=True)
    pos_after = task_b.position_embedding.weight

    assert not torch.equal(pos_before, pos_after), \
        "position_embedding did not change after warm start!"
    print(f"  PASS: warm start completed, position_embedding changed")

    print("\n" + "=" * 70)
    print("TASK B SASREC FASHION — SMOKE TEST PASSED")
    print("=" * 70)
    print()
    print("Fashion architecture summary:")
    print(f"  SharedEmbedding  : {NUM_ROWS:,} x {HIDDEN} (padding_idx={PADDING_IDX})")
    print(f"  unified_padding  : {PADDING_IDX} (≠ 0 — brand IDs start from 0)")
    print(f"  item unified IDs : {ITEM_ID_START}..{NUM_KG_ENTITIES-1} ({NUM_ITEMS_REAL:,} items)")
    print(f"  n_layers         : 4, n_heads: 4")
    print(f"  hidden_size      : {HIDDEN}, inner_size: 256")
"""
samplers_fashion.py
====================

Negative samplers for Fashion alternate learning.

NegativeSamplerKGE  — Task A (TransE): corrupts the head or tail of triples
NegativeSamplerRec  — Task B (SASRec): samples negative items in the unified space

Differences compared with B2B (samplers.py):
  - NegativeSamplerKGE: num_entities=182983 (not 24957)
    Padding (182983) is NOT a KG entity: randint(0, 182983) generates
    IDs 0..182982, therefore padding is never sampled. No fix required.
  - NegativeSamplerRec: samples from the unified item ID space (17514..182982),
    NOT from 0..num_items. In B2B, the range was 1..num_items-1 (padding=0).
    In Fashion, the range is ITEM_ID_START..NUM_KG_ENTITIES-1 (182982).
    Padding (182983) is excluded by construction from the sampling range.

File location:
    fashion_generalization/alternate_learning/models/samplers_fashion.py
"""

import torch


# =========================================================================== #
# Task A — NegativeSamplerKGE
# =========================================================================== #

class NegativeSamplerKGE:
    """
    Negative sampler for Task A (TransE).
    Randomly corrupts the head (50%) or tail (50%) of a positive triple.

    Identical to B2B — only num_entities changes at instantiation time.

    Parameters
    ----------
    num_entities : int
        Number of KG entities (does NOT include unified padding).
        For Fashion: 182983.
        randint(0, 182983) generates IDs 0..182982 — padding (182983) is excluded.
    """

    def __init__(self, num_entities: int):
        self.num_entities = num_entities

    def sample(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> tuple:
        """
        Parameters
        ----------
        h, r, t : LongTensor (B,)
            Positive triples in unified ID space.

        Returns
        -------
        h_neg, r, t_neg : LongTensor (B,)
            Negative triples with either the head or tail corrupted.
        """
        batch_size = h.size(0)
        device = h.device

        corrupt_head_mask = torch.rand(batch_size, device=device) < 0.5
        random_entities   = torch.randint(0, self.num_entities, (batch_size,), device=device)

        h_neg = torch.where(corrupt_head_mask, random_entities, h)
        t_neg = torch.where(corrupt_head_mask, t, random_entities)

        return h_neg, r, t_neg


# =========================================================================== #
# Task B — NegativeSamplerRec
# =========================================================================== #

class NegativeSamplerRec:
    """
    Negative sampler for Task B (SASRec).
    Samples negative items in the unified space while avoiding the positive item.

    CRITICAL DIFFERENCE compared with B2B:
    In B2B, RecBole items used padding=0 and item IDs 1..num_items-1,
    therefore sampling used randint(1, num_items).
    In Fashion, unified item IDs are 17514..182982 (ITEM_ID_START..182982).
    Padding (182983) is excluded by construction from the sampling range.

    Parameters
    ----------
    item_id_start : int
        First unified item ID. For Fashion: 17514.
    item_id_end : int
        Last unified item ID, inclusive. For Fashion: 182982.
        randint(item_id_start, item_id_end + 1) generates IDs 17514..182982.
    """

    def __init__(self, item_id_start: int = 17_514, item_id_end: int = 182_982):
        self.item_id_start = item_id_start
        self.item_id_end   = item_id_end
        self.num_items     = item_id_end - item_id_start + 1  # 165469

    def sample(self, pos_items: torch.Tensor, num_neg: int = 1) -> torch.Tensor:
        """
        Parameters
        ----------
        pos_items : LongTensor (B,)
            Unified IDs of positive items (test-set targets).
        num_neg : int
            Number of negatives per positive.
            If 1: returns (B,). If >1: returns (B, num_neg).

        Returns
        -------
        neg_items : LongTensor (B,) or (B, num_neg)
        """
        batch_size = pos_items.size(0)
        device     = pos_items.device

        # Sample in [item_id_start, item_id_end]
        neg_items = torch.randint(
            self.item_id_start,
            self.item_id_end + 1,
            (batch_size, num_neg),
            device=device,
        )

        # Resolve collisions with the positive item
        pos_view = pos_items.unsqueeze(1)  # (B, 1)
        collision_mask = (neg_items == pos_view)

        if collision_mask.any():
            # Shift by +1 with wrap-around in [item_id_start, item_id_end]
            colliding = neg_items[collision_mask]
            shifted = self.item_id_start + (colliding - self.item_id_start + 1) % self.num_items
            neg_items[collision_mask] = shifted

        if num_neg == 1:
            return neg_items.squeeze(1)
        return neg_items


# =========================================================================== #
# SMOKE TEST
# =========================================================================== #

if __name__ == "__main__":
    print("=" * 70)
    print("SAMPLERS FASHION — SMOKE TEST")
    print("=" * 70)

    # Real Fashion dimensions
    NUM_KG_ENTITIES = 182_983   # KG entities (0..182982), padding=182983
    ITEM_ID_START   = 17_514
    ITEM_ID_END     = 182_982
    NUM_ITEMS_REAL  = 165_469
    NUM_RELATIONS   = 3
    BATCH           = 256

    # ------------------------------------------------------------------ #
    # TEST 1: NegativeSamplerKGE
    # ------------------------------------------------------------------ #
    print("\n--- Test 1: NegativeSamplerKGE ---")
    kge_sampler = NegativeSamplerKGE(num_entities=NUM_KG_ENTITIES)

    h_pos = torch.randint(0, NUM_KG_ENTITIES, (BATCH,))
    r_pos = torch.randint(0, NUM_RELATIONS, (BATCH,))
    t_pos = torch.randint(0, NUM_KG_ENTITIES, (BATCH,))

    h_neg, r_neg, t_neg = kge_sampler.sample(h_pos, r_pos, t_pos)

    # Shape
    assert h_neg.shape == (BATCH,), f"Incorrect h_neg shape: {h_neg.shape}"
    assert t_neg.shape == (BATCH,), f"Incorrect t_neg shape: {t_neg.shape}"
    assert r_neg.shape == (BATCH,), f"Incorrect r_neg shape: {r_neg.shape}"
    print(f"  Shape: ✓")

    # Relation must not be corrupted
    assert torch.equal(r_neg, r_pos), "The relation must not be corrupted!"
    print(f"  Relation unchanged: ✓")

    # Every triple is corrupted (either head OR tail changes, not both)
    head_corrupted = (h_neg != h_pos)
    tail_corrupted = (t_neg != t_pos)
    assert (head_corrupted | tail_corrupted).all(), \
        "Some triples were not corrupted!"
    # Head and tail must not both be corrupted simultaneously
    assert not (head_corrupted & tail_corrupted).any(), \
        "Some triples have both head and tail corrupted!"
    print(f"  Exclusive head/tail corruption: ✓")

    # Approximately 50/50 distribution
    head_pct = head_corrupted.float().mean().item() * 100
    tail_pct = tail_corrupted.float().mean().item() * 100
    print(f"  Corruption distribution: head={head_pct:.1f}%, tail={tail_pct:.1f}%")
    assert 30 < head_pct < 70, f"Abnormal head corruption distribution: {head_pct:.1f}%"

    # Range: generated IDs are in [0, NUM_KG_ENTITIES-1], padding (182983) is never sampled
    assert h_neg.min().item() >= 0
    assert h_neg.max().item() <= NUM_KG_ENTITIES - 1
    assert t_neg.min().item() >= 0
    assert t_neg.max().item() <= NUM_KG_ENTITIES - 1
    # Explicitly verify that padding is never generated
    assert (h_neg == NUM_KG_ENTITIES).sum().item() == 0, \
        "Padding (182983) was sampled as a head!"
    assert (t_neg == NUM_KG_ENTITIES).sum().item() == 0, \
        "Padding (182983) was sampled as a tail!"
    print(f"  ID range [0..{NUM_KG_ENTITIES-1}], padding never sampled: ✓")
    print(f"  PASS: NegativeSamplerKGE")

    # ------------------------------------------------------------------ #
    # TEST 2: NegativeSamplerRec — 1 negative per positive
    # ------------------------------------------------------------------ #
    print("\n--- Test 2: NegativeSamplerRec (num_neg=1) ---")
    rec_sampler = NegativeSamplerRec(
        item_id_start=ITEM_ID_START,
        item_id_end=ITEM_ID_END,
    )

    # Stress test: all positives are identical (maximum collision probability)
    pos_stress = torch.full((BATCH,), fill_value=ITEM_ID_START + 100, dtype=torch.long)
    neg_stress  = rec_sampler.sample(pos_stress, num_neg=1)

    assert neg_stress.shape == (BATCH,), f"Incorrect shape: {neg_stress.shape}"
    assert (neg_stress != pos_stress).all(), "Collisions found with constant positive items!"
    assert neg_stress.min().item() >= ITEM_ID_START, \
        f"Found ID < ITEM_ID_START: {neg_stress.min().item()}"
    assert neg_stress.max().item() <= ITEM_ID_END, \
        f"Found ID > ITEM_ID_END: {neg_stress.max().item()}"
    print(f"  Shape (B,): ✓")
    print(f"  Collisions: 0 ✓")
    print(f"  Range [{ITEM_ID_START}..{ITEM_ID_END}]: ✓")

    # Test with random positive items
    pos_rand = torch.randint(ITEM_ID_START, ITEM_ID_END + 1, (BATCH,))
    neg_rand  = rec_sampler.sample(pos_rand, num_neg=1)
    assert (neg_rand != pos_rand).all(), "Collisions found with random positive items!"
    print(f"  Random positives, zero collisions: ✓")
    print(f"  PASS: NegativeSamplerRec num_neg=1")

    # ------------------------------------------------------------------ #
    # TEST 3: NegativeSamplerRec — multiple negatives per positive
    # ------------------------------------------------------------------ #
    print("\n--- Test 3: NegativeSamplerRec (num_neg=5) ---")
    NUM_NEG = 5
    neg_multi = rec_sampler.sample(pos_rand, num_neg=NUM_NEG)

    assert neg_multi.shape == (BATCH, NUM_NEG), \
        f"Incorrect shape: {neg_multi.shape}, expected ({BATCH}, {NUM_NEG})"
    pos_view = pos_rand.unsqueeze(1).expand_as(neg_multi)
    assert (neg_multi != pos_view).all(), "Collisions found in multi-negative mode!"
    assert neg_multi.min().item() >= ITEM_ID_START
    assert neg_multi.max().item() <= ITEM_ID_END
    print(f"  Shape (B, {NUM_NEG}): ✓")
    print(f"  Collisions: 0 ✓")
    print(f"  Range [{ITEM_ID_START}..{ITEM_ID_END}]: ✓")
    print(f"  PASS: NegativeSamplerRec num_neg={NUM_NEG}")

    # ------------------------------------------------------------------ #
    # TEST 4: padding (182983) is never sampled by either sampler
    # ------------------------------------------------------------------ #
    print("\n--- Test 4: padding (182983) is never sampled ---")
    PADDING = 182_983
    N_TRIALS = 10
    for _ in range(N_TRIALS):
        h_n, _, t_n = kge_sampler.sample(h_pos, r_pos, t_pos)
        assert (h_n == PADDING).sum() == 0, "KGE sampler sampled the padding!"
        assert (t_n == PADDING).sum() == 0, "KGE sampler sampled the padding!"
        neg_n = rec_sampler.sample(pos_rand, num_neg=1)
        assert (neg_n == PADDING).sum() == 0, "Recommendation sampler sampled the padding!"
    print(f"  {N_TRIALS} runs, padding never sampled by KGE or recommendation sampler: ✓")
    print(f"  PASS")

    print("\n" + "=" * 70)
    print("SAMPLERS FASHION — SMOKE TEST PASSED")
    print("=" * 70)
    print()
    print("Fashion summary:")
    print(f"  NegativeSamplerKGE : num_entities={NUM_KG_ENTITIES} → IDs 0..{NUM_KG_ENTITIES-1}")
    print(f"  NegativeSamplerRec : item IDs {ITEM_ID_START}..{ITEM_ID_END} ({NUM_ITEMS_REAL:,} items)")
    print(f"  Padding ({PADDING})   : never sampled by either sampler")
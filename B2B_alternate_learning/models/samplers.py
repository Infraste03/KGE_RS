import torch

class NegativeSamplerKGE:
    """
    Negative sampler for task A (TransE).
    Randomly corrupts the head (50%) or tail (50%) of a positive triplet.
    """
    def __init__(self, num_entities: int):
        self.num_entities = num_entities

    def sample(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> tuple:
        """
        Returns the negative triplets (h_neg, r, t_neg) by corrupting h or t.
        """
        batch_size = h.size(0)
        device = h.device

        # Randomly decide which triplets to corrupt: True for head, False for tail
        corrupt_head_mask = torch.rand(batch_size, device=device) < 0.5

        # Sample random entities for corruption
        random_entities = torch.randint(0, self.num_entities, (batch_size,), device=device)

        # Where the mask is True (corrupt h), keep t. Otherwise (corrupt t), replace t.
        h_neg = torch.where(corrupt_head_mask, random_entities, h)
        # Where the mask is True (corrupt h), keep h. Otherwise (corrupt t), replace t.
        t_neg = torch.where(corrupt_head_mask, t, random_entities)
        # Ensure that we don't accidentally sample the same entity as the original h or t
        return h_neg, r, t_neg


"""new code, if i want to sample multiple negatives for task B, i can use this class"""
class NegativeSamplerRec:
    """
    Negative sampler for task B (SASRec).
    Randomly samples one or more negative items, avoiding the positive item.
    """
    def __init__(self, num_items: int, padding_idx: int = None):
        # num_items represents the maximum dictionary
        self.num_items = num_items
        self.padding_idx = padding_idx

    def sample(self, pos_items: torch.Tensor, num_neg: int = 1) -> torch.Tensor:
        """
        Produces a tensor of negative items.
        If num_neg == 1 returns shape(B, ) for backward compatibility.
        If num_neg > 1 returns shape(B, num_neg).
        """
        batch_size = pos_items.size(0)
        device = pos_items.device

        # Create a tensor (B, num_neg) of random values
        #  logic padding:
        neg_items = torch.randint(0, self.num_items, (batch_size, num_neg), device=device)
        if self.padding_idx is not None:
             while (neg_items == self.padding_idx).any():
                  mask = (neg_items == self.padding_idx)
                  neg_items[mask] = torch.randint(0, self.num_items, (mask.sum().item(),), device=device)

        # pos_items originally is (B,). We take it to (B, 1) so we can broadcast
        pos_items_view = pos_items.unsqueeze(1)

        # Collision checking: Boolean mask (B, num_neg)
        collision_mask = (neg_items == pos_items_view)

        if collision_mask.any():
            # We shift or change by resolving loops to prevent hit pddding
            shifted = (neg_items[collision_mask] + 1) % self.num_items
            if self.padding_idx is not None:
                 pad_mask = (shifted == self.padding_idx)
                 if pad_mask.any():
                      shifted[pad_mask] = (shifted[pad_mask] + 1) % self.num_items
            neg_items[collision_mask] = shifted

        # For backward compatibility, if only 1 negative is required, we flatten to (B, )
        if num_neg == 1:
            return neg_items.squeeze(1)

        return neg_items

# =========================================================================== #
# SMOKE TEST
# =========================================================================== #

if __name__ == "__main__":
    print("=" * 70)
    print("SAMPLERS SMOKE TEST")
    print("=" * 70)

    NUM_ENTITIES = 24957
    NUM_ITEMS = 21135
    BATCH = 256

    # ---------------------------------------------------------
    # TEST TASK A: KGE Sampler
    # ---------------------------------------------------------
    print("\n--- Test Task A: KGE Sampler ---")
    kge_sampler = NegativeSamplerKGE(num_entities=NUM_ENTITIES)

    # Data dummy
    h_pos = torch.randint(0, NUM_ENTITIES, (BATCH,))
    r_pos = torch.randint(0, 5, (BATCH,))
    t_pos = torch.randint(0, NUM_ENTITIES, (BATCH,))

    h_neg, r_neg, t_neg = kge_sampler.sample(h_pos, r_pos, t_pos)

    # 1. Shapes Check
    assert h_neg.shape == h_pos.shape, "error shape on h_neg"
    assert t_neg.shape == t_pos.shape, "error shape on t_neg"
    # 2. Check corruption: r_neg must be identical to r_pos
    assert torch.equal(r_neg, r_pos), "The sampler must not corrupt the relationship"

    # 3. Corruption distribution (about 50-50)
    # Corrupt head where h_neg!= h_pos, corrupted queue where t_neg != t_pos
    head_corrupted = (h_neg != h_pos).sum().item()
    tail_corrupted = (t_neg != t_pos).sum().item()
    print(f"  Shape output corretta: {h_neg.shape}")
    print(f"  head corrupted: {head_corrupted}/{BATCH} ({head_corrupted/BATCH*100:.1f}%)")
    print(f"  tail corrupted: {tail_corrupted}/{BATCH} ({tail_corrupted/BATCH*100:.1f}%)")
    # At least one head or tail must have been altered in each row.
    assert (h_neg != h_pos).logical_or(t_neg != t_pos).all(), "Some triplets were not corrupted at all!"
    print("  KGE Sampler: PASS")


    # ---------------------------------------------------------
    # TEST TASK B: Rec Sampler
    # ---------------------------------------------------------
    print("\n--- Test Task B: Rec Sampler ---")
    rec_sampler = NegativeSamplerRec(num_items=NUM_ITEMS)

    # I simulate that by infamous coincidence, the batch of pos_items is all done by the same ID (e.g. 5)
    # to stress the resolution of the collision.
    pos_items = torch.full((BATCH,), fill_value=5)
    neg_items = rec_sampler.sample(pos_items)
    # 1. Shape Check
    assert neg_items.shape == pos_items.shape, "Shape errore on neg_items"
    # 2. Collision Check: Absolutely zero elements equal to the positives
    collisions = (neg_items == pos_items).sum().item()
    assert collisions == 0, f"Found {collisions} collision with the positives!"
    # 3. Range Check (Niente index out of bounds, no zero)
    assert neg_items.min().item() >= 1, "Found ID <= 0 in SASRec negatives!"
    assert neg_items.max().item() < NUM_ITEMS, "Found ID over-bound in SASRec negatives!"
    print(f"  Shape output correct: {neg_items.shape}")
    print(f"  found collisions: {collisions}")
    print(f"  Min ID generated: {neg_items.min().item()} | Max ID generated: {neg_items.max().item()}")
    print("  Rec Sampler: PASS")
    pos_items_random = torch.randint(1, NUM_ITEMS, (BATCH,))
    neg_items_random = rec_sampler.sample(pos_items_random)
    assert (neg_items_random == pos_items_random).sum().item() == 0, "collisions with random pos!"
    print(f"  Test with random pos: PASS")

    print("\n" + "=" * 70)
    print("ALL SAMPLER TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
"""
data_loaders_fashion.py
========================

DataLoaders for Fashion alternate learning.

Two functions:
  build_kg_dataloader_fashion  — Task A: reads kg_train.tsv and converts triples
                                 into unified IDs
  build_rec_dataloader_fashion — Task B: materializes RecBole training data
                                 into native PyTorch tensors

Differences compared with B2B (data_loaders.py):
  - build_kg_dataloader_fashion: identical to B2B; only the source KG entity
    types differ (item/category/brand instead of item/machine/model/client)
  - build_rec_dataloader_fashion: RecBole padding (ID=0) is mapped to
    UNIFIED_PADDING=182983 through the r2u tensor
    (instead of 0 as in B2B).
    The MockRecBoleLoader used in the smoke test is consistent with the
    way the loop consumes RecBole batches.

File location:
    fashion_generalization/alternate_learning/data_loaders_fashion.py
"""

import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Fashion constants
UNIFIED_PADDING  = 182_983
ITEM_ID_START    = 17_514
ITEM_ID_END      = 182_982


def build_kg_dataloader_fashion(
    kg_file_path: str,
    entity_to_unified: dict,
    relation_to_id: dict,
    batch_size: int = 1024,
    shuffle: bool = True,
) -> DataLoader:
    """
    Read kg_train.tsv and return a DataLoader containing (h, r, t) triples
    already mapped to the Fashion unified ID space.

    This is identical to the B2B implementation. Only the entity prefixes
    in the source KG differ, but this function does not need to know them
    because it receives mappings already constructed by load_kg_fashion
    and unified_id_space_fashion.

    Parameters
    ----------
    kg_file_path : str
        Path to kg_train.tsv (columns: head, relation, tail).
    entity_to_unified : dict
        Mapping entity_string -> unified_id. From kg['entity_to_id'].
    relation_to_id : dict
        Mapping relation_string -> int. From kg['relation_to_id'].
    batch_size : int
        Default 1024. Reduce to 512 if OOM occurs.
    shuffle : bool
        True for training, False for debugging.

    Returns
    -------
    DataLoader yielding batches (h, r, t) of LongTensor.
    """
    logger.info(f"  Reading KG from {kg_file_path}...")
    df = pd.read_csv(kg_file_path, sep='\t')

    # Remove header if it appears as the first data row
    if df.iloc[0]['head'] == 'head':
        df = df.iloc[1:].reset_index(drop=True)

    # Filter triples containing entities or relations not present in the mappings
    valid_mask = (
        df['head'].isin(entity_to_unified) &
        df['tail'].isin(entity_to_unified) &
        df['relation'].isin(relation_to_id)
    )
    df_valid = df[valid_mask]
    dropped = len(df) - len(df_valid)
    if dropped > 0:
        logger.warning(
            f"  Dropped {dropped} triples containing unknown entities/relations."
        )

    heads = torch.tensor(
        [entity_to_unified[h] for h in df_valid['head'].values], dtype=torch.long
    )
    rels = torch.tensor(
        [relation_to_id[r] for r in df_valid['relation'].values], dtype=torch.long
    )
    tails = torch.tensor(
        [entity_to_unified[t] for t in df_valid['tail'].values], dtype=torch.long
    )

    # Verify ID ranges
    assert heads.min() >= 0 and heads.max() < UNIFIED_PADDING, \
        f"Head IDs out of range: [{heads.min()}, {heads.max()}]"
    assert tails.min() >= 0 and tails.max() < UNIFIED_PADDING, \
        f"Tail IDs out of range: [{tails.min()}, {tails.max()}]"
    assert (heads == UNIFIED_PADDING).sum() == 0, \
        "Padding found among head IDs!"
    assert (tails == UNIFIED_PADDING).sum() == 0, \
        "Padding found among tail IDs!"

    dataset = TensorDataset(heads, rels, tails)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True
    )
    logger.info(
        f"  Fashion KG DataLoader: {len(dataset):,} triples, "
        f"{len(dataloader)} batches (batch_size={batch_size})"
    )
    return dataloader


def build_rec_dataloader_fashion(
    recbole_train_dataloader,
    recbole_to_unified_tensor: torch.Tensor,
    batch_size: int = 256,
    shuffle: bool = True,
) -> DataLoader:
    """
    Materialize the RecBole training DataLoader into a native PyTorch TensorDataset.

    Iterates once over the RecBole DataLoader, translates RecBole IDs into unified
    IDs through the r2u tensor, and produces a reusable DataLoader for training.

    DIFFERENCE COMPARED WITH B2B:
    RecBole padding (ID=0) is mapped to UNIFIED_PADDING=182983 by the r2u tensor.
    In B2B, unified padding was 0; here it is 182983.

    TaskBSASRec uses (item_seq != 182983) for the attention mask, therefore
    no additional transformation is required here once the IDs have been mapped.

    Parameters
    ----------
    recbole_train_dataloader : RecBole DataLoader
        RecBole training DataLoader. Each batch is an Interaction object
        accessed through batch['item_id_list'], batch['item_length'],
        and batch['item_id'].
    recbole_to_unified_tensor : LongTensor (num_items_recbole,)
        Translation tensor from RecBole internal ID -> unified ID.
        It is constructed from the RecBole-to-unified mapping provided by
        UnifiedIdSpace.
        Position 0 (RecBole padding) -> 182983 (unified padding).
    batch_size : int
        Batch size for the materialized DataLoader. Default 256.
    shuffle : bool
        True for training.

    Returns
    -------
    DataLoader yielding batches
    (item_seq_unified, item_seq_len, pos_items_unified).

        item_seq_unified : LongTensor (B, 50) — sequences in unified IDs
        item_seq_len     : LongTensor (B,)    — actual sequence lengths
        pos_items_unified: LongTensor (B,)    — target items in unified IDs
    """
    logger.info("  Materializing RecBole training DataLoader...")

    all_seqs     = []
    all_lens     = []
    all_pos      = []

    for batch in recbole_train_dataloader:
        # RecBole training DataLoader normally yields Interaction objects
        # (key-based access rather than tuple indexing).
        if isinstance(batch, (tuple, list)):
            # Compatibility fallback: if a tuple is returned, use its first element
            interaction = batch[0]
        else:
            interaction = batch

        seq_rec = interaction['item_id_list']   # (B, T) RecBole IDs
        seq_len = interaction['item_length']    # (B,)
        pos_rec = interaction['item_id']        # (B,) RecBole IDs

        # Vectorized RecBole -> unified translation
        seq_uni = recbole_to_unified_tensor[seq_rec]   # (B, T) unified IDs
        pos_uni = recbole_to_unified_tensor[pos_rec]   # (B,) unified IDs

        all_seqs.append(seq_uni.cpu())
        all_lens.append(seq_len.cpu())
        all_pos.append(pos_uni.cpu())

    t_seqs = torch.cat(all_seqs, dim=0)   # (N, T)
    t_lens = torch.cat(all_lens, dim=0)   # (N,)
    t_pos  = torch.cat(all_pos,  dim=0)   # (N,)

    # Post-materialization verification
    # Unified padding (182983) is expected inside sequences as padding positions
    # but must NEVER appear as a positive target
    assert (t_pos == UNIFIED_PADDING).sum() == 0, \
        "Padding (182983) appears as pos_item — error in the r2u tensor!"

    # Positive targets must all correspond to real items (17514..182982)
    assert t_pos.min().item() >= ITEM_ID_START, \
        f"pos_item < ITEM_ID_START: min={t_pos.min().item()}"
    assert t_pos.max().item() <= ITEM_ID_END, \
        f"pos_item > ITEM_ID_END: max={t_pos.max().item()}"

    # Sequences must contain only real items or padding
    valid_seq_mask = (t_seqs == UNIFIED_PADDING) | \
                     ((t_seqs >= ITEM_ID_START) & (t_seqs <= ITEM_ID_END))
    assert valid_seq_mask.all(), \
        "Sequences contain IDs that are neither real items nor padding — error in the r2u tensor!"

    logger.info(
        f"  Fashion Rec DataLoader: {len(t_seqs):,} materialized sequences"
    )
    logger.info(
        f"  pos_item range: [{t_pos.min().item()}, {t_pos.max().item()}] "
        f"(expected [{ITEM_ID_START}, {ITEM_ID_END}])"
    )
    padding_in_seqs = (t_seqs == UNIFIED_PADDING).sum().item()
    logger.info(
        f"  Padding positions (182983) in sequences: {padding_in_seqs:,} "
        f"(expected > 0; these are empty positions)"
    )

    dataset = TensorDataset(t_seqs, t_lens, t_pos)
    safe_drop_last = len(dataset) > batch_size
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=safe_drop_last,
    )
    logger.info(
        f"  DataLoader: {len(dataloader)} batches (batch_size={batch_size}, "
        f"drop_last={safe_drop_last})"
    )
    return dataloader


# =========================================================================== #
# SMOKE TEST
# =========================================================================== #

if __name__ == "__main__":
    print("=" * 70)
    print("DATA LOADERS FASHION — SMOKE TEST")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # TEST 1: build_kg_dataloader_fashion
    # ------------------------------------------------------------------ #
    print("\n--- Test 1: build_kg_dataloader_fashion ---")

    dummy_kg_path = "dummy_kg_fashion_test.tsv"
    with open(dummy_kg_path, "w") as f:
        f.write("head\trelation\ttail\n")
        f.write("item_A\tcompatible_with\titem_B\n")
        f.write("item_B\tbelongs_to\tcategory_X\n")
        f.write("item_C\tbelongs_to_brand\tbrand_Y\n")
        f.write("item_X\tunknown_rel\titem_Y\n")  # must be dropped

    # Fashion-like mapping with realistic brand/category/item offsets
    dummy_e2u = {
        "brand_Y":     5,
        "category_X":  17511,
        "item_A":      17514,
        "item_B":      17515,
        "item_C":      17516,
    }
    dummy_r2i = {
        "compatible_with":  0,
        "belongs_to":       1,
        "belongs_to_brand": 2,
    }

    kg_loader = build_kg_dataloader_fashion(
        kg_file_path=dummy_kg_path,
        entity_to_unified=dummy_e2u,
        relation_to_id=dummy_r2i,
        batch_size=2,
        shuffle=False,
    )

    batch_h, batch_r, batch_t = next(iter(kg_loader))

    assert batch_h.shape == (2,), f"Incorrect h shape: {batch_h.shape}"
    assert batch_r.shape == (2,), f"Incorrect r shape: {batch_r.shape}"
    assert batch_t.shape == (2,), f"Incorrect t shape: {batch_t.shape}"
    assert batch_h.dtype == torch.long
    assert batch_r.dtype == torch.long
    assert batch_t.dtype == torch.long

    # Verify that values correspond to the correct unified IDs
    assert batch_h[0].item() == dummy_e2u["item_A"], \
        f"Incorrect head: {batch_h[0].item()}"
    assert batch_r[0].item() == dummy_r2i["compatible_with"], \
        f"Incorrect relation: {batch_r[0].item()}"
    assert batch_t[0].item() == dummy_e2u["item_B"], \
        f"Incorrect tail: {batch_t[0].item()}"

    print(f"  Batch h={batch_h.tolist()}, r={batch_r.tolist()}, t={batch_t.tolist()} ✓")
    print(f"  Triple with unknown_rel dropped: ✓")
    print(f"  IDs are correct unified IDs: ✓")
    print(f"  PASS: build_kg_dataloader_fashion")

    os.remove(dummy_kg_path)

    # ------------------------------------------------------------------ #
    # TEST 2: build_rec_dataloader_fashion — translation and padding
    # ------------------------------------------------------------------ #
    print("\n--- Test 2: build_rec_dataloader_fashion ---")

    # Mock yielding Interaction-like objects (dicts, as in real RecBole usage)
    # rather than tuples
    class MockRecBoleTrainLoader:
        def __iter__(self):
            # Batch 1: 3 users, sequences with RecBole padding (0)
            yield {
                'item_id_list': torch.tensor([
                    [1, 2, 0, 0, 0],   # user 0: 2 items + 3 padding
                    [3, 4, 5, 0, 0],   # user 1: 3 items + 2 padding
                    [2, 0, 0, 0, 0],   # user 2: 1 item + 4 padding
                ]),
                'item_length': torch.tensor([2, 3, 1]),
                'item_id':     torch.tensor([5, 6, 3]),   # target RecBole IDs
            }

            # Batch 2: 2 users
            yield {
                'item_id_list': torch.tensor([
                    [6, 5, 4, 3, 0],
                    [1, 2, 3, 4, 5],
                ]),
                'item_length': torch.tensor([4, 5]),
                'item_id':     torch.tensor([2, 1]),
            }

    # Fashion-like r2u tensor:
    # RecBole ID 0 (padding) -> 182983 (unified padding)
    # RecBole IDs 1..6 -> realistic unified item IDs
    r2u = torch.tensor([
        182_983,   # 0 = RecBole padding -> unified padding
        17_514,    # 1 -> first item
        17_515,    # 2
        17_516,    # 3
        17_517,    # 4
        17_518,    # 5
        17_519,    # 6
    ], dtype=torch.long)

    rec_loader = build_rec_dataloader_fashion(
        recbole_train_dataloader=MockRecBoleTrainLoader(),
        recbole_to_unified_tensor=r2u,
        batch_size=3,
        shuffle=False,
    )

    batch_seq, batch_len, batch_pos = next(iter(rec_loader))

    # Shape
    assert batch_seq.shape == (3, 5), f"Incorrect seq shape: {batch_seq.shape}"
    assert batch_len.shape == (3,),   f"Incorrect len shape: {batch_len.shape}"
    assert batch_pos.shape == (3,),   f"Incorrect pos shape: {batch_pos.shape}"
    print(f"  Shape (3, 5) / (3,) / (3,): ✓")

    # Correct translation: RecBole 0 -> 182983, RecBole 1 -> 17514, etc.
    assert batch_seq[0, 0].item() == 17_514, \
        f"Incorrect seq[0,0]: {batch_seq[0,0].item()} (expected 17514)"
    assert batch_seq[0, 1].item() == 17_515, \
        f"Incorrect seq[0,1]: {batch_seq[0,1].item()} (expected 17515)"
    assert batch_seq[0, 2].item() == 182_983, \
        f"Incorrect seq[0,2]: {batch_seq[0,2].item()} (expected 182983 = padding)"

    print(f"  RecBole -> unified translation correct: ✓")
    print(f"  RecBole padding (0) -> 182983: ✓")

    # Positive items translated correctly
    assert batch_pos[0].item() == 17_518, \
        f"Incorrect pos[0]: {batch_pos[0].item()} (expected 17518, RecBole 5)"
    assert batch_pos[1].item() == 17_519, \
        f"Incorrect pos[1]: {batch_pos[1].item()} (expected 17519, RecBole 6)"
    print(f"  pos_items translated correctly: ✓")

    # No positive target is padding
    assert (batch_pos == 182_983).sum() == 0, \
        "Padding found among pos_items!"
    print(f"  pos_items contain no padding: ✓")

    # Sequences contain only real items (>=17514) or padding (182983)
    valid = (batch_seq == 182_983) | (batch_seq >= 17_514)
    assert valid.all(), "Sequences contain invalid IDs!"
    print(f"  Sequences contain only real items or padding 182983: ✓")
    print(f"  PASS: build_rec_dataloader_fashion")

    # ------------------------------------------------------------------ #
    # TEST 3: compatibility with alternate_loop
    # ------------------------------------------------------------------ #
    print("\n--- Test 3: compatibility with alternate_loop ---")

    # The loop performs:
    # item_seq, item_seq_len, pos_items = (x.to(device) for x in batch_b)
    # Verify that each batch is a tuple of three tensors
    # (TensorDataset guarantees this structure)
    for batch in rec_loader:
        assert len(batch) == 3, \
            f"Batch must contain 3 elements, found {len(batch)}"

        seq, lns, pos = batch

        assert seq.dtype == torch.long, "item_seq must be LongTensor"
        assert lns.dtype == torch.long, "item_seq_len must be LongTensor"
        assert pos.dtype == torch.long, "pos_items must be LongTensor"

        # Verify that item_seq_len matches the number of non-padding positions
        for b in range(seq.shape[0]):
            non_pad = (seq[b] != 182_983).sum().item()
            assert non_pad == lns[b].item(), \
                f"item_seq_len[{b}]={lns[b].item()} != non-padding count={non_pad}"
        break

    print(f"  Batch is a tuple (seq, len, pos) of LongTensor: ✓")
    print(f"  item_seq_len matches non-padding positions: ✓")
    print(f"  PASS: compatible with alternate_loop")

    print("\n" + "=" * 70)
    print("DATA LOADERS FASHION — SMOKE TEST PASSED")
    print("=" * 70)
    print()
    print("Fashion summary:")
    print(f"  KG DataLoader   : triples (h,r,t) in unified IDs [0..182982]")
    print(f"  Rec DataLoader  : sequences in unified IDs [{ITEM_ID_START}..{ITEM_ID_END}]")
    print(f"                    RecBole padding (0) -> {UNIFIED_PADDING}")
    print(f"  Compatibility   : batch = (item_seq, item_seq_len, pos_items), as in B2B")
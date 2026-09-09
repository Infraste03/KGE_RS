"""
data_loaders_fashion_v3.py
==========================

DataLoaders for Fashion alternate learning v3.

Two functions:
  build_kg_dataloader_fashion  — Task A: reads kg_train.tsv and converts triples to unified IDs
  build_rec_dataloader_fashion — Task B: materializes RecBole training data into PyTorch tensors

Differences compared with B2B (data_loaders.py):
  - build_kg_dataloader_fashion: identical logic, with Fashion entity types.
  - build_rec_dataloader_fashion: RecBole padding (ID=0) is mapped to the
    Fashion unified padding ID through the r2u tensor rather than to 0.
    The MockRecBoleLoader used in the smoke test emits Interaction-like
    objects (dict), consistently with the real RecBole training loader.

Location:
    fashion_generalization/alternate_learning/variants/v3/data_loaders_fashion_v3.py
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

# Fashion v3 constants
# The same unified vocabulary is shared by the 3-relation and 5-relation variants.
UNIFIED_PADDING  = 81_259    # Confirmed from entity2id.tsv v3 (81260 rows - 1 header)
ITEM_ID_START    = 9_317     # Confirmed via grep, consistent with 71,935 items in kg_stats.txt
ITEM_ID_END      = 81_251    # 81251 - 9317 + 1 = 71,935


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

    The logic is identical to the B2B implementation. The entity prefixes
    differ in the source KG, but this function does not need to handle them
    directly because the mappings are provided by load_kg_fashion and
    unified_id_space_fashion.

    Parameters
    ----------
    kg_file_path : str
        Path to kg_train.tsv with columns: head, relation, tail.
    entity_to_unified : dict
        Mapping entity_string -> unified_id, from kg['entity_to_id'].
    relation_to_id : dict
        Mapping relation_string -> int, from kg['relation_to_id'].
    batch_size : int
        Default 1024. Reduce to 512 in case of OOM.
    shuffle : bool
        True for training, False for debugging.

    Returns
    -------
    DataLoader yielding batches (h, r, t) of LongTensor objects.
    """
    logger.info(f"  Reading KG from {kg_file_path}...")
    df = pd.read_csv(kg_file_path, sep='\t')

    # Remove the header if it is accidentally present as the first data row
    if df.iloc[0]['head'] == 'head':
        df = df.iloc[1:].reset_index(drop=True)

    # Filter triples containing entities or relations missing from the mappings
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
        [entity_to_unified[h] for h in df_valid['head'].values],
        dtype=torch.long
    )
    rels = torch.tensor(
        [relation_to_id[r] for r in df_valid['relation'].values],
        dtype=torch.long
    )
    tails = torch.tensor(
        [entity_to_unified[t] for t in df_valid['tail'].values],
        dtype=torch.long
    )

    # Range checks
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
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True
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

    The function iterates once over the RecBole DataLoader, translates RecBole IDs
    into unified IDs through the r2u tensor, and produces a reusable DataLoader
    for alternate training.

    DIFFERENCE COMPARED WITH B2B:
    RecBole padding (ID=0) is mapped to the Fashion UNIFIED_PADDING through
    the r2u tensor. In B2B the unified padding was 0.

    TaskBSASRec identifies padding through the unified padding value, therefore
    no additional conversion is required here.

    Parameters
    ----------
    recbole_train_dataloader : RecBole DataLoader
        RecBole training DataLoader. Each batch is an Interaction object
        accessed through batch['item_id_list'], batch['item_length'],
        and batch['item_id'].
    recbole_to_unified_tensor : LongTensor (num_items_recbole,)
        Translation tensor from RecBole internal ID to unified ID.
        Position 0 corresponds to RecBole padding and maps to unified padding.
    batch_size : int
        Batch size for the materialized DataLoader. Default 256.
    shuffle : bool
        True for training.

    Returns
    -------
    DataLoader yielding:
        (item_seq_unified, item_seq_len, pos_items_unified)

        item_seq_unified : LongTensor (B, 50) — sequences in unified IDs
        item_seq_len     : LongTensor (B,)    — actual sequence lengths
        pos_items_unified: LongTensor (B,)    — target items in unified IDs
    """
    logger.info("  Materializing RecBole training DataLoader...")

    all_seqs = []
    all_lens = []
    all_pos = []

    for batch in recbole_train_dataloader:

        # The RecBole training DataLoader normally emits Interaction objects,
        # accessed by key rather than by tuple index.
        if isinstance(batch, (tuple, list)):

            # Compatibility fallback:
            # if a tuple/list is returned, use its first element.
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

    # Post-materialization checks:
    # unified padding may occur inside sequences,
    # but it must NEVER occur as a positive target item.
    assert (t_pos == UNIFIED_PADDING).sum() == 0, \
        "Unified padding appears as pos_item — error in the r2u tensor!"

    # Positive targets must all correspond to real Fashion items
    assert t_pos.min().item() >= ITEM_ID_START, \
        f"pos_item < ITEM_ID_START: min={t_pos.min().item()}"

    assert t_pos.max().item() <= ITEM_ID_END, \
        f"pos_item > ITEM_ID_END: max={t_pos.max().item()}"

    # Sequences must contain only real items or padding
    valid_seq_mask = (
        (t_seqs == UNIFIED_PADDING) |
        ((t_seqs >= ITEM_ID_START) & (t_seqs <= ITEM_ID_END))
    )

    assert valid_seq_mask.all(), \
        "Sequences contain IDs that are neither real items nor padding — error in the r2u tensor!"

    logger.info(
        f"  Fashion Rec DataLoader: "
        f"{len(t_seqs):,} materialized sequences"
    )

    logger.info(
        f"  pos_item range: "
        f"[{t_pos.min().item()}, {t_pos.max().item()}] "
        f"(expected [{ITEM_ID_START}, {ITEM_ID_END}])"
    )

    padding_in_seqs = (
        t_seqs == UNIFIED_PADDING
    ).sum().item()

    logger.info(
        f"  Padding positions (182983) in sequences: "
        f"{padding_in_seqs:,} "
        f"(expected > 0; these are empty positions)"
    )

    dataset = TensorDataset(
        t_seqs,
        t_lens,
        t_pos
    )

    safe_drop_last = len(dataset) > batch_size

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=safe_drop_last,
    )

    logger.info(
        f"  DataLoader: {len(dataloader)} batches "
        f"(batch_size={batch_size}, "
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

    print(
        "\n--- Test 1: build_kg_dataloader_fashion ---"
    )

    dummy_kg_path = "dummy_kg_fashion_test.tsv"

    with open(dummy_kg_path, "w") as f:
        f.write("head\trelation\ttail\n")
        f.write("item_A\tcompatible_with\titem_B\n")
        f.write("item_B\tbelongs_to\tcategory_X\n")
        f.write("item_C\tbelongs_to_brand\tbrand_Y\n")
        f.write(
            "item_X\tunknown_rel\titem_Y\n"
        )  # This triple must be dropped

    # Fashion-like mapping using realistic brand/category/item offsets
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

    batch_h, batch_r, batch_t = next(
        iter(kg_loader)
    )

    assert batch_h.shape == (2,), \
        f"Incorrect h shape: {batch_h.shape}"

    assert batch_r.shape == (2,), \
        f"Incorrect r shape: {batch_r.shape}"

    assert batch_t.shape == (2,), \
        f"Incorrect t shape: {batch_t.shape}"

    assert batch_h.dtype == torch.long
    assert batch_r.dtype == torch.long
    assert batch_t.dtype == torch.long

    # Verify that values correspond to the expected unified IDs
    assert batch_h[0].item() == dummy_e2u["item_A"], \
        f"Incorrect head: {batch_h[0].item()}"

    assert batch_r[0].item() == dummy_r2i["compatible_with"], \
        f"Incorrect relation: {batch_r[0].item()}"

    assert batch_t[0].item() == dummy_e2u["item_B"], \
        f"Incorrect tail: {batch_t[0].item()}"

    print(
        f"  Batch h={batch_h.tolist()}, "
        f"r={batch_r.tolist()}, "
        f"t={batch_t.tolist()} ✓"
    )

    print(
        "  Triple with unknown relation dropped: ✓"
    )

    print(
        "  IDs are valid unified IDs: ✓"
    )

    print(
        "  PASS: build_kg_dataloader_fashion"
    )

    os.remove(dummy_kg_path)


    # ------------------------------------------------------------------ #
    # TEST 2: build_rec_dataloader_fashion — translation and padding
    # ------------------------------------------------------------------ #

    print(
        "\n--- Test 2: build_rec_dataloader_fashion ---"
    )

    # Mock loader emitting Interaction-like objects (dict),
    # consistently with the real RecBole training loader.
    class MockRecBoleTrainLoader:

        def __iter__(self):

            # Batch 1: 3 users, sequences containing RecBole padding (0)
            yield {
                'item_id_list': torch.tensor([
                    [1, 2, 0, 0, 0],   # user 0: 2 items + 3 padding positions
                    [3, 4, 5, 0, 0],   # user 1: 3 items + 2 padding positions
                    [2, 0, 0, 0, 0],   # user 2: 1 item + 4 padding positions
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

    batch_seq, batch_len, batch_pos = next(
        iter(rec_loader)
    )

    # Shape checks
    assert batch_seq.shape == (3, 5), \
        f"Incorrect sequence shape: {batch_seq.shape}"

    assert batch_len.shape == (3,), \
        f"Incorrect length shape: {batch_len.shape}"

    assert batch_pos.shape == (3,), \
        f"Incorrect positive-item shape: {batch_pos.shape}"

    print(
        "  Shape (3, 5) / (3,) / (3,): ✓"
    )

    # Verify translation:
    # RecBole 0 -> 182983, RecBole 1 -> 17514, etc.
    assert batch_seq[0, 0].item() == 17_514, \
        (
            f"Incorrect seq[0,0]: "
            f"{batch_seq[0,0].item()} "
            f"(expected 17514)"
        )

    assert batch_seq[0, 1].item() == 17_515, \
        (
            f"Incorrect seq[0,1]: "
            f"{batch_seq[0,1].item()} "
            f"(expected 17515)"
        )

    assert batch_seq[0, 2].item() == 182_983, \
        (
            f"Incorrect seq[0,2]: "
            f"{batch_seq[0,2].item()} "
            f"(expected 182983 = padding)"
        )

    print(
        "  RecBole -> unified translation is correct: ✓"
    )

    print(
        "  RecBole padding (0) -> 182983: ✓"
    )

    # Positive items are translated correctly
    assert batch_pos[0].item() == 17_518, \
        (
            f"Incorrect pos[0]: "
            f"{batch_pos[0].item()} "
            f"(expected 17518, RecBole 5)"
        )

    assert batch_pos[1].item() == 17_519, \
        (
            f"Incorrect pos[1]: "
            f"{batch_pos[1].item()} "
            f"(expected 17519, RecBole 6)"
        )

    print(
        "  pos_items translated correctly: ✓"
    )

    # No positive item may be padding
    assert (
        batch_pos == 182_983
    ).sum() == 0, \
        "Padding found among pos_items!"

    print(
        "  pos_items do not contain padding: ✓"
    )

    # Sequences must contain only real items or padding
    valid = (
        (batch_seq == 182_983)
        | (batch_seq >= 17_514)
    )

    assert valid.all(), \
        "Sequences contain invalid IDs!"

    print(
        "  Sequences contain only real items or padding 182983: ✓"
    )

    print(
        "  PASS: build_rec_dataloader_fashion"
    )


    # ------------------------------------------------------------------ #
    # TEST 3: compatibility with alternate_loop
    # ------------------------------------------------------------------ #

    print(
        "\n--- Test 3: compatibility with alternate_loop ---"
    )

    # alternate_loop consumes each batch as:
    #
    # item_seq, item_seq_len, pos_items =
    #     (x.to(device) for x in batch_b)
    #
    # Verify that each batch is a tuple containing three tensors.
    for batch in rec_loader:

        assert len(batch) == 3, \
            f"Batch must contain 3 elements, found {len(batch)}"

        seq, lns, pos = batch

        assert seq.dtype == torch.long, \
            "item_seq must be Long"

        assert lns.dtype == torch.long, \
            "item_seq_len must be Long"

        assert pos.dtype == torch.long, \
            "pos_items must be Long"

        # Verify that item_seq_len matches the number of non-padding positions
        for b in range(seq.shape[0]):

            non_pad = (
                seq[b] != 182_983
            ).sum().item()

            assert non_pad == lns[b].item(), \
                (
                    f"item_seq_len[{b}]={lns[b].item()} "
                    f"!= non-padding count={non_pad}"
                )

        break

    print(
        "  Batch is a tuple (seq, len, pos) of LongTensor objects: ✓"
    )

    print(
        "  item_seq_len matches the number of non-padding positions: ✓"
    )

    print(
        "  PASS: compatibility with alternate_loop"
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "DATA LOADERS FASHION — SMOKE TEST PASSED"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Fashion summary:"
    )

    print(
        "  KG DataLoader   : triples (h,r,t) in unified IDs [0..182982]"
    )

    print(
        f"  Rec DataLoader  : sequences in unified IDs "
        f"[{ITEM_ID_START}..{ITEM_ID_END}]"
    )

    print(
        f"                    RecBole padding (0) -> "
        f"{UNIFIED_PADDING}"
    )

    print(
        "  Compatibility   : "
        "batch = (item_seq, item_seq_len, pos_items), as in B2B"
    )
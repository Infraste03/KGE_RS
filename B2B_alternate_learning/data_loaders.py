import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def build_kg_dataloader(
    kg_file_path: str,
    entity_to_unified: dict,
    relation_to_id: dict,
    batch_size: int = 1024,
    shuffle: bool = True
) -> DataLoader:
    """
    Reads the Knowledge Graph TSV file and returns a PyTorch DataLoader
    whose triplets (h, r, t) are given mapped in the Unified ID Space.
    """
    logger.info(f"Loading KG from {kg_file_path}...")
    df = pd.read_csv(kg_file_path, sep='\t', names=['head', 'relation', 'tail'])

    if df.iloc[0]['head'] == 'head':
        df = df.iloc[1:]
    valid_rows = df['head'].isin(entity_to_unified) & \
                 df['tail'].isin(entity_to_unified) & \
                 df['relation'].isin(relation_to_id)
    df_valid = df[valid_rows]
    dropped = len(df) - len(df_valid)
    if dropped > 0:
        logger.warning(f"Attention: {dropped} triples dropped due to unknown entities/relations.")
    heads = torch.tensor([entity_to_unified[h] for h in df_valid['head'].values], dtype=torch.long)
    rels = torch.tensor([relation_to_id[r] for r in df_valid['relation'].values], dtype=torch.long)
    tails = torch.tensor([entity_to_unified[t] for t in df_valid['tail'].values], dtype=torch.long)

    dataset = TensorDataset(heads, rels, tails)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    logger.info(f"KG DataLoader created: {len(dataset)} triples, {len(dataloader)} batches.")
    return dataloader


def build_rec_dataloader(
    recbole_dataloader,
    recbole_to_unified_tensor: torch.Tensor,
    batch_size: int = 256,
    shuffle: bool = True
) -> DataLoader:
    """
    Cycles through a complete RecBole DataLoader (training set), extracts the sequences
    and targets, converts the RecBole IDs to Unified IDs via the mapping tensor,
    and pre-materializes everything in a native PyTorch TensorDataset.
    """
    logger.info("Materialization of the RecBole Dataloader into native PyTorch tensors...")
    device = recbole_to_unified_tensor.device
    all_seqs = []
    all_lens = []
    all_pos_items = []
    for batch in recbole_dataloader:
        seq_rec = batch['item_id_list'].to(device)
        seq_len = batch['item_length'].to(device)
        pos_rec = batch['item_id'].to(device)
        seq_uni = recbole_to_unified_tensor[seq_rec]
        pos_uni = recbole_to_unified_tensor[pos_rec]
        all_seqs.append(seq_uni.cpu().view(-1, seq_uni.shape[-1])) # (Batch, Seq_Len)
        all_lens.append(seq_len.cpu().view(-1))                    #(Batch,)
        all_pos_items.append(pos_uni.cpu().view(-1)) 


    t_seqs = torch.cat(all_seqs, dim=0)
    t_lens = torch.cat(all_lens, dim=0)
    t_pos_items = torch.cat(all_pos_items, dim=0)
    dataset = TensorDataset(t_seqs, t_lens, t_pos_items)
    safe_drop_last = len(dataset) > batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=safe_drop_last)
    if not safe_drop_last:
        logger.warning(
            f"drop_last=False: dataset ({len(dataset)} Examples) smaller than batch_size ({batch_size}). "
            f"Consider reducing batch_size."
        )
    else:
        logger.info(f"Rec DataLoader created: {len(dataset)} sequences, {len(dataloader)} batches.")
    return dataloader

# =========================================================================== #
# SMOKE TEST
# =========================================================================== #

if __name__ == "__main__":
    print("=" * 70)
    print("DATALOADERS SMOKE TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # TEST TASK A: KG DataLoader
    # ---------------------------------------------------------
    print("\n--- Test Task A: KGE DataLoader ---")
    dummy_kg_path = "dummy_kg_test.tsv"
    with open(dummy_kg_path, "w") as f:
        f.write("head\trelation\ttail\n")
        f.write("item_A\tcompatibile\titem_B\n")
        f.write("item_B\tcompra\titem_C\n")
        f.write("item_D\tsimile\titem_A\n")
        f.write("item_X\tignoto\titem_Y\n")
    dummy_e2u = {"item_A": 10, "item_B": 11, "item_C": 12, "item_D": 13}
    dummy_r2i = {"compatibile": 0, "compra": 1, "simile": 2}

    kg_loader = build_kg_dataloader(
        kg_file_path=dummy_kg_path,
        entity_to_unified=dummy_e2u,
        relation_to_id=dummy_r2i,
        batch_size=2,
        shuffle=False
    )
    batch_h, batch_r, batch_t = next(iter(kg_loader))

    assert batch_h.shape == (2,), "Shape head error"
    assert batch_r.shape == (2,), "Shape relation error"
    assert batch_h.dtype == torch.long, "tensor should be long type"
    print(f"  PASS: Batch extracted successfully -> h:{batch_h}, r:{batch_r}, t:{batch_t}")
    os.remove(dummy_kg_path)


    # ---------------------------------------------------------
    # TEST TASK B: Rec DataLoader
    # ---------------------------------------------------------
    print("\n--- Test Task B: Rec DataLoader ---")

    # Creazione di un mock del Dataloader recbole
    class MockRecBoleLoader:
        def __iter__(self):
            interaction = {
                'item_id_list': torch.tensor([[1, 2, 0], [3, 0, 0]]),
                'item_length': torch.tensor([2, 1]),
                'item_id': torch.tensor([3, 4])
            }
            yield (interaction, None, None, None)

            interaction2 = {
                'item_id_list': torch.tensor([[4, 1, 2], [2, 0, 0]]),
                'item_length': torch.tensor([3, 1]),
                'item_id': torch.tensor([1, 3])
            }
            yield (interaction2, None, None, None)

    mapping_tensor = torch.tensor([0, 101, 102, 103, 104])

    rec_loader = build_rec_dataloader(
        recbole_dataloader=MockRecBoleLoader(),
        recbole_to_unified_tensor=mapping_tensor,
        batch_size=2,
        shuffle=False
    )
    batch_seq, batch_len, batch_pos = next(iter(rec_loader))

    assert batch_seq.shape == (2, 3), "Shape sequence error"
    assert batch_len.shape == (2,), "Shape lengths error"
    assert batch_pos.shape == (2,), "Shape pos_items error"

    assert batch_pos.tolist() == [103, 104], f"Failure in mapping RecBole->Unified, got {batch_pos.tolist()}"
    print(f"  PASS: The mapping worked (The unified positive items are {batch_pos.tolist()})")
    print(f"  PASS: Mapped sequences: {batch_seq.tolist()}")

    print("\n" + "=" * 70)
    print("ALL DATALOADER TESTS PASSED SUCCESSFULLY! ")
    print("=" * 70)
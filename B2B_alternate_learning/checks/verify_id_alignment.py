"""
Verify ID alignment across the three components of Step 4:

  1. our load_kg.py mapping (entity_string -> int)
  2. PyKEEN's mapping used in Step 2 to train TransE
  3. RecBole's mapping used in Step 3 to train SASRec

We need ALL THREE to be aligned (or at least translatable) before we can
implement hard sharing. Otherwise the warm-started weights would be loaded
into the wrong rows, and the model would silently produce garbage.

----------------------------------------------------------------------------
Where to put this file:
    B2B_alternate_learning/checks/verify_id_alignment.py
----------------------------------------------------------------------------
"""

import os
import sys
import torch
import pandas as pd
import numpy as np

# Make load_kg importable from the parent directory
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, PARENT_DIR)
from load_kg import load_kg

# --------------------------------------------------------------------------- #
# Paths (adjust to your project structure)
# --------------------------------------------------------------------------- #

# Project Root Directory
BASE_DIR = os.path.dirname(PARENT_DIR)

# Our KG TSV files
KG_TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "kg_train.tsv")
KG_VALID_PATH = os.path.join(BASE_DIR, "data", "processed", "taskA_valid.tsv")
KG_TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "taskA_test.tsv")

# RecBole files
RECBOLE_DATASET_NAME = "b2b_data"
RECBOLE_DATA_PATH = os.path.join(BASE_DIR, "dataset")  # parent folder of b2b_data/
RECBOLE_TRAIN_INTER = os.path.join(BASE_DIR, "dataset", "b2b_data", "b2b_data.train.inter")
RECBOLE_VALID_INTER = os.path.join(BASE_DIR, "dataset", "b2b_data", "b2b_data.valid.inter")
RECBOLE_TEST_INTER = os.path.join(BASE_DIR, "dataset", "b2b_data", "b2b_data.test.inter")


# --------------------------------------------------------------------------- #
# 1. Helper: extract item IDs from RecBole .inter files (without RecBole)
# --------------------------------------------------------------------------- #

def extract_item_ids_from_inter_files(*inter_paths):
    """
    Read RecBole .inter files and return the set of unique item IDs (as strings).

    .inter files use this format (tab-separated):
        user_id:token<tab>item_id:token<tab>timestamp:float
        12345           67890              1577836800
        ...

    The first row is the header. The item_id column contains the original token
    (e.g. '67890' or 'item_1234' depending on how the dataset was prepared).
    """
    all_items = set()
    for path in inter_paths:
        if not os.path.exists(path):
            print(f"  WARNING: file not found, skipping: {path}")
            continue
        df = pd.read_csv(path, sep='\t')
        # The item column may be named 'item_id:token' or just 'item_id'
        item_col = None
        for col in df.columns:
            if col.startswith('item_id'):
                item_col = col
                break
        if item_col is None:
            raise ValueError(f"No 'item_id' column found in {path}")
        all_items.update(df[item_col].astype(str).unique())
    return all_items


# --------------------------------------------------------------------------- #
# 2. Helper: try to extract RecBole's internal mapping
# --------------------------------------------------------------------------- #

def get_recbole_item_mapping():
    """
    Reconstruct RecBole's item_id -> internal_id mapping.

    RecBole assigns internal IDs in this order:
      - id 0 is reserved for padding
      - subsequent ids are assigned in the order items are encountered
        in the .inter files (after sorting/filtering steps)

    The most robust way to get the mapping is to actually load the dataset
    via RecBole. This requires RecBole to be installed.
    """
    try:
        from recbole.config import Config
        from recbole.data import create_dataset
    except ImportError:
        print("  RecBole not available; falling back to manual reconstruction.")
        return None

    config = Config(
        model='SASRec',
        dataset=RECBOLE_DATASET_NAME,
        config_dict={
            'data_path': RECBOLE_DATA_PATH,
            'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
            'MAX_ITEM_LIST_LENGTH': 300,
            'embedding_size': 64,
            'show_progress': False,
            'loss_type': 'BPR',
            'train_neg_sample_args': None
        }
    )
    dataset = create_dataset(config)

    # In RecBole, the field2token_id dict maps field name -> {token_string -> internal_int_id}
    item_field = config['ITEM_ID_FIELD']
    item_token_to_id = dataset.field2token_id[item_field]
    return item_token_to_id


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    print("=" * 70)
    print("VERIFY ID ALIGNMENT")
    print("=" * 70)

    # ------------------------------------------------------------------- #
    # STEP 1: load our KG mapping
    # ------------------------------------------------------------------- #
    print("\n[1/4] Loading our KG mapping (load_kg.py) ...")
    kg = load_kg(
        train_path=KG_TRAIN_PATH,
        valid_path=KG_VALID_PATH,
        test_path=KG_TEST_PATH,
        verbose=False,
    )

    # Extract just the item entities and their (string -> our_id) mapping
    our_item_to_id = {
        e: i for e, i in kg['entity_to_id'].items() if e.startswith('item_')
    }
    # Strip the 'item_' prefix so we can compare with RecBole/CSV-style IDs
    our_item_strings_stripped = {e[5:] for e in our_item_to_id.keys()}

    print(f"  Our KG has {len(our_item_to_id):,} item entities")
    print(f"  Sample: {list(our_item_to_id.items())[:3]}")

    # ------------------------------------------------------------------- #
    # STEP 2: extract RecBole's item set from .inter files
    # ------------------------------------------------------------------- #
    print("\n[2/4] Extracting items from RecBole .inter files ...")
    recbole_items_raw = extract_item_ids_from_inter_files(
        RECBOLE_TRAIN_INTER, RECBOLE_VALID_INTER, RECBOLE_TEST_INTER,
    )
    print(f"  RecBole .inter files contain {len(recbole_items_raw):,} unique items")
    print(f"  Sample: {list(recbole_items_raw)[:3]}")

    # ------------------------------------------------------------------- #
    # STEP 3: compare the two item sets
    # ------------------------------------------------------------------- #
    print("\n[3/4] Comparing the two item sets ...")

    in_both = our_item_strings_stripped & recbole_items_raw
    only_in_kg = our_item_strings_stripped - recbole_items_raw
    only_in_recbole = recbole_items_raw - our_item_strings_stripped

    print(f"  Items in BOTH KG and RecBole: {len(in_both):,}")
    print(f"  Items in KG but NOT in RecBole: {len(only_in_kg):,}")
    print(f"  Items in RecBole but NOT in KG: {len(only_in_recbole):,}")

    if only_in_kg:
        print(f"  Sample 'only in KG': {list(only_in_kg)[:5]}")
    if only_in_recbole:
        print(f"  Sample 'only in RecBole': {list(only_in_recbole)[:5]}")

    # ------------------------------------------------------------------- #
    # STEP 4: try to extract RecBole's internal mapping
    # ------------------------------------------------------------------- #
    print("\n[4/4] Extracting RecBole's internal id mapping ...")
    recbole_mapping = get_recbole_item_mapping()

    if recbole_mapping is None:
        print("  Could not extract RecBole mapping (RecBole not available or error).")
        print("  We can still proceed if we reconstruct the mapping manually later.")
    else:
        print(f"  RecBole's internal mapping has {len(recbole_mapping):,} items")
        print(f"  (this includes the padding token at id=0)")

        # Show some samples
        sorted_by_id = sorted(recbole_mapping.items(), key=lambda x: x[1])
        print(f"  First 5 entries (by internal id):")
        for token, iid in sorted_by_id[:5]:
            print(f"    '{token}' -> {iid}")

    # ------------------------------------------------------------------- #
    # FINAL DIAGNOSTIC
    # ------------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    if len(only_in_kg) == 0 and len(only_in_recbole) == 0:
        print("PERFECT alignment: KG and RecBole share the exact same item set.")
        print("We can build hard sharing without any item filtering.")
    elif len(only_in_kg) > 0 and len(only_in_recbole) == 0:
        print(f"PARTIAL alignment:")
        print(f"  - All RecBole items are in the KG (good).")
        print(f"  - {len(only_in_kg)} KG items are NOT in RecBole sequences.")
        print(f"  - These are items that no client ever bought during the test period.")
        print(f"  - For the alternate learning: SharedEmbedding will hold them,")
        print(f"    Task A will train them via 'compatible_with', Task B will not see them.")
        print(f"  - This is FINE: it's actually one of the values of using KG.")
    elif len(only_in_recbole) > 0 and len(only_in_kg) == 0:
        print(f"PROBLEMATIC alignment:")
        print(f"  - {len(only_in_recbole)} RecBole items are NOT in the KG.")
        print(f"  - SASRec knows about these items but TransE never saw them.")
        print(f"  - In SharedEmbedding they would be random and never trained by Task A.")
        print(f"  - We need to decide: include them in KG or remove from sequences.")
    else:
        print(f"COMPLEX alignment:")
        print(f"  - {len(only_in_kg)} items only in KG")
        print(f"  - {len(only_in_recbole)} items only in RecBole")
        print(f"  - Investigate before proceeding.")

    print()
    print("Next steps depend on what we found above.")


if __name__ == "__main__":
    main()
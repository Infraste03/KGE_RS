"""
verify_id_alignment_fashion.py
================================
Check ID alignment between:
  1. load_kg_fashion.py (entity_string -> int, primary IDs)
  2. PyKEEN (used for TransE HPO fashion)
  3. RecBole (used for SASRec HPO fashion)


"""

import os
import sys
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
BASE_DIR   = os.path.abspath(os.path.join(PARENT_DIR, ".."))
sys.path.insert(0, PARENT_DIR)

from load_kg_fashion import load_kg_fashion


KG_TRAIN = os.path.join(BASE_DIR, "data", "processed", "kg_train.tsv")
KG_VALID = os.path.join(BASE_DIR, "data", "processed", "taskA_valid.tsv")
KG_TEST  = os.path.join(BASE_DIR, "data", "processed", "taskA_test.tsv")

INTER_PATH       = os.path.join(BASE_DIR, "data", "recbole", "fashion", "fashion.inter")
RECBOLE_DATA_PATH = os.path.join(BASE_DIR, "data", "recbole")


def extract_item_tokens_from_inter(inter_path: str) -> set:
    """Extracts item tokens from the unique.inter file (LOO done by RecBole internally)."""
    if not os.path.exists(inter_path):
        raise FileNotFoundError(f"File not found: {inter_path}")
    df = pd.read_csv(inter_path, sep='\t')
    item_col = next((c for c in df.columns if c.startswith('item_id')), None)
    if item_col is None:
        raise ValueError(f"No item_id column in {inter_path}")
    return set(df[item_col].astype(str).unique())


def get_recbole_mapping(data_path: str) -> dict:
    """Reconstructs the mapping item_token -> internal_id of RecBole."""
    from recbole.config import Config
    from recbole.data import create_dataset

    config = Config(
        model='SASRec',
        dataset='fashion',
        config_dict={
            'data_path':       data_path,
            'USER_ID_FIELD':   'user_id',
            'ITEM_ID_FIELD':   'item_id',
            'TIME_FIELD':      'timestamp',
            'load_col':        {'inter': ['user_id', 'item_id', 'timestamp']},
            'field_separator': "\t",
            'eval_args': {
                'split':    {'LS': 'valid_and_test'},
                'order':    'TO',
                'group_by': 'user',
                'mode':     {'valid': 'full', 'test': 'full'},
            },
            'MAX_ITEM_LIST_LENGTH': 50,
            'embedding_size':       64,
            'show_progress':        False,
            'train_neg_sample_args': None,
        }
    )
    dataset = create_dataset(config)
    return dict(dataset.field2token_id[config['ITEM_ID_FIELD']])



def main():
    print("=" * 70)
    print("VERIFY ID ALIGNMENT — FASHION")
    print("=" * 70)

    # ---- 1. KG mapping ----
    print("\n[1/4] Loading KG mapping (load_kg_fashion.py)...")
    kg = load_kg_fashion(KG_TRAIN, KG_VALID, KG_TEST, verbose=False)

    our_item_to_id = {
        e: i for e, i in kg['entity_to_id'].items() if e.startswith('item_')
    }
    our_items_stripped = {e[len('item_'):] for e in our_item_to_id}
    print(f"  Item in KG         : {len(our_item_to_id):,}")
    print(f"  Sample            : {list(our_item_to_id.items())[:3]}")

    # ---- 2. RecBole item set dal file .inter ----
    print("\n[2/4] Extracting items from fashion.inter...")
    recbole_tokens = extract_item_tokens_from_inter(INTER_PATH)
    print(f"  Unique items in .inter: {len(recbole_tokens):,}")
    print(f"  Sample            : {list(recbole_tokens)[:3]}")

    # ---- 3.  KG vs .inter ----
    print("\n[3/4] KG vs RecBole .inter comparison...")
    in_both        = our_items_stripped & recbole_tokens
    only_in_kg     = our_items_stripped - recbole_tokens
    only_in_recbole = recbole_tokens - our_items_stripped

    print(f"  in both        : {len(in_both):,}")
    print(f"  only in KG        : {len(only_in_kg):,}")
    print(f"  only in RecBole    : {len(only_in_recbole):,}")
    if only_in_kg:
        print(f"  Sample only in KG   : {list(only_in_kg)[:5]}")
    if only_in_recbole:
        print(f"  Sample only in Rec  : {list(only_in_recbole)[:5]}")

    # ---- 4. RecBole internal mapping ----
    print("\n[4/4] RecBole internal mapping reconstruction...")
    recbole_mapping = get_recbole_mapping(RECBOLE_DATA_PATH)
    print(f"  total Entries (with [PAD]): {len(recbole_mapping):,}")

    sorted_by_id = sorted(recbole_mapping.items(), key=lambda x: x[1])
    print(f"  first 5 for internal id:")
    for token, iid in sorted_by_id[:5]:
        print(f"    '{token}' -> {iid}")

    # ---- Diagnostic ----
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    if len(only_in_kg) == 0 and len(only_in_recbole) == 0:
        print("PERFECT alignment: KG and RecBole share exactly the same")
        print("set of items. Hard sharing without additional filters.")
    elif len(only_in_kg) > 0 and len(only_in_recbole) == 0:
        print(f"PARTIAL alignment (expected for some datasets):")
        print(f" - All RecBole items are in the KG (OK).")
        print(f" - {len(only_in_kg)} KG items do not appear in RecBole sequences.")
        print(f" - They will only be trained by Task A.")
    elif len(only_in_recbole) > 0 and len(only_in_kg) == 0:
        print(f"PROBLEMATIC: {len(only_in_recbole)} RecBole items are not in KG.")
        print(f" -> TransE has never seen them. To be investigated.")
    else:
        print(f"COMPLEX CASE: {len(only_in_kg)} KG only, {len(only_in_recbole)} RecBole only.")
        print(f" -> Investigate before proceeding.")


if __name__ == "__main__":
    main()
"""
check_recbole_mapping_fashion.py
=================================

Verify that the RecBole config used in unified_id_space_fashion.py
ALWAYS produces the same mapping item_token -> internal_id.

The script rebuilds the mapping TWICE with the same config and
check that they are identical. This ensures that the config is "fixed"
and it won't accidentally change in data_loaders.py.


"""

import os
import sys

THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
PARENT    = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, PARENT)

BASE = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
DATA_PATH = os.path.join(BASE, "data", "recbole")


def _build_mapping(data_path: str) -> dict:
    """Rebuilds the item_token -> internal_id mapping using RecBole."""
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
    item_field = config['ITEM_ID_FIELD']
    return dict(dataset.field2token_id[item_field])


def main():
    print("=" * 70)
    print("CHECK RECBOLE MAPPING DETERMINISM — FASHION")
    print("=" * 70)

    print("\n[1/3] First mapping construction...")
    mapping_1 = _build_mapping(DATA_PATH)
    print(f"  Total Entries (including [PAD]): {len(mapping_1):,}")

    print("\n[2/3] Second mapping construction (same config)...")
    mapping_2 = _build_mapping(DATA_PATH)
    print(f"  Total Entries (including [PAD]): {len(mapping_2):,}")

    print("\n[3/3] Comparison...")

    assert len(mapping_1) == len(mapping_2), \
        f"Length different: {len(mapping_1)} vs {len(mapping_2)}"

    keys_1 = set(mapping_1.keys())
    keys_2 = set(mapping_2.keys())
    assert keys_1 == keys_2, \
        f"Keys different: only in 1={keys_1 - keys_2}, only in 2={keys_2 - keys_1}"

    mismatches = {
        k: (mapping_1[k], mapping_2[k])
        for k in mapping_1
        if mapping_1[k] != mapping_2[k]
    }
    assert len(mismatches) == 0, \
        f"ID different for {len(mismatches)} tokens. Sample: {list(mismatches.items())[:5]}"

    assert mapping_1.get('[PAD]', None) == 0, \
        f"[PAD] should have id=0, got {mapping_1.get('[PAD]')}"

    items_sorted = sorted(
        [(k, v) for k, v in mapping_1.items() if k != '[PAD]'],
        key=lambda x: x[1]
    )
    print(f"\n  First 5 item (per internal id):")
    for token, iid in items_sorted[:5]:
        print(f"    '{token}' -> {iid}")

    print(f"\n  Last 5 item (per internal id):")
    for token, iid in items_sorted[-5:]:
        print(f"    '{token}' -> {iid}")

    print("\n" + "=" * 70)
    print("CHECK PASSED — mapping is deterministic")
    print("=" * 70)
    print(f"  Total real items (excluding PAD): {len(mapping_1) - 1:,}")
    print()
    print("IMPORTANT: The config used here must be IDENTICAL")
    print("to the one in unified_id_space_fashion.py and in data_loaders_fashion.py.")
    print("If you change even a single parameter, the mapping changes and the")
    print("translation from unified <-> recbole breaks silently.")


if __name__ == "__main__":
    main()
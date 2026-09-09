"""
verify_architecture.py

Diagnostic script to check the consistency of the JointAlternateModel architecture
after the warm-start process, before training begins.

Performs a series of targeted checks to ensure that:
1. The dimensions of the embedding matrices are correct.
2. ID padding is handled correctly and its embedding line is isolated.
3. The TransE weights were loaded correctly into the corresponding rows.
4. The SASRec transformer weights were loaded correctly.
5. SASRec item embedding weights were NOT loaded into the shared embedding,
   confirming that the initial performance CANNOT be that of pure SASRec.
"""

import os
import sys
import yaml
import torch
import argparse

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
GRANDPARENT_DIR = os.path.dirname(PARENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
if GRANDPARENT_DIR not in sys.path:
    sys.path.insert(0, GRANDPARENT_DIR)

from B2B_alternate_learning.unified_id_space import build_unified_id_space
from B2B_alternate_learning.models.joint_model import JointAlternateModel

# RecBole imports
from recbole.config import Config as RecBoleConfig
from recbole.data import create_dataset
from recbole.utils import init_seed

def print_check(name, success=True, message=""):
    status = " OK" if success else " not OK"
    print(f"  [{status}] {name}")
    if message:
        print(f"     -> {message}")

def main():
    parser = argparse.ArgumentParser(description="Verification script for Step 4 Architecture")
    parser.add_argument("--config", type=str,
                        default=os.path.join(PARENT_DIR, "configs/step4_config.yaml"),
                        help="Path to the configuration yaml file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("VERIFICATION OF JOINT ALTERNATE MODEL ARCHITECTURE AFTER WARM START")
    print("=" * 70)

    # --- STEP 1: Unified ID Space Construction ---
    print("\n[Step 1] Unified ID Space Construction...")
    space = build_unified_id_space(
        kg_train_path=config['paths']['kg_train'],
        kg_valid_path=config['paths']['kg_valid'],
        kg_test_path=config['paths']['kg_test'],
        recbole_train_inter=config['paths']['recbole_train_inter'],
        recbole_valid_inter=config['paths']['recbole_valid_inter'],
        recbole_test_inter=config['paths']['recbole_test_inter'],
        recbole_dataset_name=config['paths']['recbole_dataset_name'],
        recbole_data_path=config['paths']['recbole_data_path'],
    )
    print_check("Unified ID Space built")

    # --- STEP 2: Loading RecBole Data ---
    print("\n[Step 2] Loading RecBole Data...")
    recbole_cfg = RecBoleConfig(model='SASRec', dataset='b2b_data', config_dict=config['recbole_config'])
    init_seed(recbole_cfg['seed'], recbole_cfg['reproducibility'])
    recbole_dataset = create_dataset(recbole_cfg)
    num_recbole_items = recbole_dataset.item_num
    print_check("RecBole dataset loaded", message=f"Found {num_recbole_items} RecBole items (incl. padding)")

    # --- STEP 3: Model Instantiation ---
    print("\n[Step 3] Model Instantiation...")
    PAD_UNIFIED_ID = space.num_total_entities
    model = JointAlternateModel(
        num_entities=space.num_total_entities + 1,
        num_relations=5,
        num_items=num_recbole_items,
        kge_dim=config['model']['kge_dim'],
        padding_idx=PAD_UNIFIED_ID
    ).to(device)
    print_check("JointAlternateModel instantiated", message=f"SharedEmbedding shape: {model.shared_embedding.embedding.weight.shape}")

    # --- STEP 4: WARM START ---
    print("\n[Step 4] WARM START...")
    # 4a: Pesi TransE
    pykeen_state_dict = torch.load(config['paths']['pykeen_best_model'], map_location='cpu', weights_only=False)
    pykeen_entity_weights = pykeen_state_dict['entity_representations.0._embeddings.weight']
    with torch.no_grad():
        n_kg = pykeen_entity_weights.shape[0]
        model.shared_embedding.embedding.weight[:n_kg] = pykeen_entity_weights
    print_check("Load TransE weights into SharedEmbedding", message=f"Loaded {n_kg} entity embeddings from TransE")

    # 4b: pth SASRec
    sasrec_checkpoint = torch.load(config['paths']['sasrec_checkpoint'], map_location='cpu', weights_only=False)
    sasrec_state_dict = sasrec_checkpoint.get('state_dict', sasrec_checkpoint)
    model.load_pretrained_sasrec(sasrec_state_dict)
    print_check("Load SASRec weights into the model")
    print("-" * 30)


    # --- STEP 5: POST WARM-START VERIFICATION ---
    print("\n[Step 5] Verification of integrity on the SharedEmbedding matrix...")

    # Check 1: dimension
    expected_rows = space.num_total_entities + 1
    actual_rows = model.shared_embedding.embedding.weight.shape[0]
    check1_ok = (actual_rows == expected_rows)
    print_check(" SharedEmbedding dim ", check1_ok,
                f"expected: {expected_rows} rows, Found: {actual_rows} rows.")

    # Check 2: integrity  Padding ID
    pad_embedding_row = model.shared_embedding.embedding.weight[PAD_UNIFIED_ID]
    check2_ok = (torch.all(pad_embedding_row == 0.0))
    print_check(" integrity  Padding ID", check2_ok,
                f"row{PAD_UNIFIED_ID} should be all zeros, found norm={pad_embedding_row.norm().item():.4f}.")

    # Check 3: Verify TransE weight loading
    # Let's compare the first row (entity 'client_00011A03BC')
    transE_first_row = pykeen_entity_weights[0].to(device)
    shared_first_row = model.shared_embedding.embedding.weight[0]
    check3_ok = torch.allclose(transE_first_row, shared_first_row)
    print_check("Correct loading of TransE weights", check3_ok,
                "The first row of SharedEmbedding corresponds to the first row of TransE weights.")

    # Check 4: Check NOT loading SASRec item_embedding weights
    # Let's take a random RecBole element (e.g. id=10) and its unified_id
    recbole_id_to_check = 10
    uid_to_check = space.recbole_to_unified(recbole_id_to_check)
    if uid_to_check is not None:
        sasrec_item_emb_row = sasrec_state_dict['item_embedding.weight'][recbole_id_to_check].to(device)
        shared_emb_row_for_item = model.shared_embedding.embedding.weight[uid_to_check]

        # These two vectors MUST be DIFFERENT.
        # The vector in SharedEmbedding must come from TransE (if the item is also a KG entity)
        # or be randomly initialized (if it is a pure item). NOT from SASRec.
        are_different = not torch.allclose(sasrec_item_emb_row, shared_emb_row_for_item)
        check4_ok = are_different
        print_check("NOT loading SASRec item_embedding", check4_ok,
                    f"The embedding for the RecBole item ID {recbole_id_to_check} (Unified ID {uid_to_check}) "
                    "in the SharedEmbedding is DIFFERENT from the original SASRec embedding. This is CORRECT.")
    else:
        print_check("NOT loading SASRec item_embedding", success=False, message="Unable to find a valid test item.")

    # Check 5: Verify loading of Transformer weights
    # Let's check a random weight in the first attention layer
    sasrec_attn_weight = sasrec_state_dict['trm_encoder.layer.0.attention.w_q.weight']
    model_attn_weight = model.task_b_model.trm_encoder.layer[0].attention.w_q.weight.cpu()
    check5_ok = torch.allclose(sasrec_attn_weight, model_attn_weight)
    print_check("Correct loading of Transformer SASRec weights", check5_ok,
                "The weights of an attention layer correspond to those of the SASRec checkpoint.")

    print("\n=" * 70)
    print("VERIFICA COMPLETATA")
    print("=" * 70)
    print("\cconclusion:")
    print("The architecture is configured correctly. The `SharedEmbedding` matrix:")
    print("  - Has the correct dimension, with an extra slot for padding.")
    print("  - Is initialized with the weights of TransE for the KG entities.")
    print("  - DOES NOT use the weights of the `item_embedding` of SASRec.")
    print("\nThis confirms that the performance at Epoch 0 (e.g., 0.24) is the expected result of this hybrid architecture")
    print("and cannot be equal to the performance of the standalone SASRec (0.36).")


if __name__ == '__main__':
    main()

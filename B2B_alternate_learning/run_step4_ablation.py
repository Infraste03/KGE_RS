"""
run_step4_ablation.py :  Orchestrator Phase 4 (Alternative Learning) for the
KG ABLATION VARIANTS (B2B).

Identical to run_step4.py, with two differences:
    1. A --variant argument selects which ablated variant to use.
       The FILTERED kg_train and pre-trained TransE weights (warm start)
       are automatically resolved by the paths already produced by
       generate_ablation_plan.py / generate_ablation_plan_extra.py and
       run_transe_ablation_batch.py - no handwritten paths,
       no risk of typo.
    2. Each variant writes to its own and separate output folder
       (hpc_step4_results/step4_ablation_<variant>_<scheduling><run_tag>/),
       so the 6 configurations (including the full one already done) are not
       they ever touch each other.

Tutto il resto -- avvio a caldo SASRec, DataLoader Task B, pianificazione,
riprendi il checkpoint, salvataggio best_model.pt / best_model_on_test.pt --
e' invariato rispetto a run_step4.py.

Come si lancia (una variante per volta, lo sfatch fara' il loop):
    python run_step4_ablation.py --config configs/step4_config.yaml --variant loo_no_compatible_with
    python run_step4_ablation.py --config configs/step4_config.yaml --variant loo_no_instance_of --seed 1
    python run_step4_ablation.py --config configs/step4_config.yaml --variante mix1_keep_compatible_with_owns --epochs 30

If a variant has already been completed (best_metrics.json present),
the script stops immediately without touching anything. To still make a run again
completed, pass --force.
"""

import os
import sys
import argparse
import yaml
import json
import random as _random
import numpy as np
import logging
import torch


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(THIS_DIR, "models")
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from unified_id_space import build_unified_id_space
from joint_model import JointAlternateModel
from data_loaders import build_kg_dataloader, build_rec_dataloader
from alternate_loop import AlternateTrainer
from eval_utils import evaluate_task_b

# RecBole imports
from recbole.config import Config as RecBoleConfig
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ABLATION_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "ablation" / "ablation_manifest.json"
ABLATION_TRANSE_ROOT = PROJECT_ROOT / "results" / "ablation_transe"
ABLATION_KGTRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "ablation"

SELECTED_VARIANTS = [
    "loo_no_compatible_with",
    "loo_no_instance_of",
    "mix1_keep_compatible_with_owns",
    "mix3_keep_bought_compatible_with",
    "targeted_triple_cw_bought_instance",
]


def resolve_ablation_paths(variant_name: str) -> dict:
    """
    Given the name of a variant, it returns:
        - kg_train_ablation: path to the FILTERED kg_train of that variant
        - pykeen_best_model: path to the pre-trained TransE weights (warm start) of that variant
    Read from the existing manifest and output structure, to avoid
    to duplicate/retranscribe path by hand.
    """
    if variant_name not in SELECTED_VARIANTS:
        raise ValueError(
            f"variant '{variant_name}' not in the list of selected variants. "
            f"Available: {SELECTED_VARIANTS}"
        )

    with open(ABLATION_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if variant_name not in manifest:
        raise ValueError(
            f"variant '{variant_name}' not found in {ABLATION_MANIFEST_PATH}. "
            f"Available keys: {list(manifest.keys())}"
        )

    kg_train_filename = os.path.basename(manifest[variant_name]["kg_train_ablation"])
    kg_train_ablation = os.path.join(ABLATION_KGTRAIN_DIR, kg_train_filename)
    pykeen_best_model = os.path.join(
        ABLATION_TRANSE_ROOT, variant_name, f"best_model_{variant_name}.pt"
    )

    if not os.path.exists(kg_train_ablation):
        raise FileNotFoundError(f"kg_train filtered not found: {kg_train_ablation}")
    if not os.path.exists(pykeen_best_model):
        raise FileNotFoundError(f"TransE weights (warm start) not found: {pykeen_best_model}")

    return {
        "kg_train_ablation": kg_train_ablation,
        "pykeen_best_model": pykeen_best_model,
    }


# =========================================================================== #
# CONFIG and LOGGING
# =========================================================================== #

def load_config():
    """Reads the YAML file, applies the CLI overrides, and resolves the paths of the ablation variant."""
    parser = argparse.ArgumentParser(description="Step 4 - Alternate Learning Training (ABLATION)")
    parser.add_argument("--config", type=str, default="configs/step4_config.yaml",
                        help="Path to the YAML configuration file (same as used for the full version)")
    parser.add_argument("--variant", type=str, required=True,
                        choices=SELECTED_VARIANTS,
                        help="Name of the ablation variant to use (filtered kg_train + warm start)")
    parser.add_argument("--scheduling", type=str,
                        choices=["one_to_one", "epoch", "adaptive"],
                        help="Override of the training.scheduling field in the YAML")
    parser.add_argument("--epochs", type=int,
                        help="Override of the number of epochs")
    parser.add_argument("--run-tag", type=str,
                        help="Suffix optional to distinguish output/trial in the same scheduling")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed (for the multi-seed runs)")
    parser.add_argument("--force", action="store_true",
                        help="run even if the variant has already been completed (best_metrics.json exists)")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override CLI standard
    if args.scheduling:
        config['training']['scheduling'] = args.scheduling
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.run_tag:
        config['training']['run_tag'] = args.run_tag
    if args.seed is not None:
        config['recbole_config']['seed'] = args.seed

    ablation_paths = resolve_ablation_paths(args.variant)
    config['paths']['kg_train_filtered'] = ablation_paths['kg_train_ablation']
    config['paths']['pykeen_best_model'] = ablation_paths['pykeen_best_model']

    config['_ablation_variant'] = args.variant
    config['_force'] = args.force

    return config


def setup_logging(output_dir):
    """Configure logger to file + stdout."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'run.log')

    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%d-%b %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# =========================================================================== #
# MAIN
# =========================================================================== #

def main():

    config = load_config()
    variant = config['_ablation_variant']

    run_tag = config['training'].get('run_tag', '')
    out_folder = os.path.join(
        "hpc_step4_results", f"step4_ablation_{variant}_{config['training']['scheduling']}{run_tag}"
    )


    best_metrics_path = os.path.join(out_folder, "best_metrics.json")
    if os.path.exists(best_metrics_path) and not config['_force']:
        print(f"[SKIP] variant '{variant}' already completed: {best_metrics_path} exists already.")
        print(f"       Use --force if you want to run it anyway (overwrites).")
        sys.exit(0)

    logger = setup_logging(out_folder)

    logger.info("=" * 70)
    logger.info("STEP 4 - ALTERNATE LEARNING TRAINING (ABLATION)")
    logger.info("=" * 70)
    logger.info(f"variant ablation : {variant}")
    logger.info(f"kg_train (original, vocabulary)     : {config['paths']['kg_train']}")
    logger.info(f"kg_train (filtered, Task A DataLoader) : {config['paths']['kg_train_filtered']}")
    logger.info(f"pth TransE (warm start)              : {config['paths']['pykeen_best_model']}")
    logger.info(f"Output dir: {out_folder}")
    logger.info(f"Scheduling: {config['training']['scheduling']}")
    logger.info(f"Epochs: {config['training']['epochs']}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    hpo_trial = os.environ.get('HPO_TRIAL', None)
    if hpo_trial is not None:
        logger.info(f"HPO Trial: {hpo_trial}")
    logger.info("Hyperparameters (fixed, taken from the full model for controlled comparison):")
    logger.info(f"  learning_rate    : {config['training'].get('learning_rate', 'N/A')}")
    logger.info(f"  batch_size_sasrec: {config['training'].get('batch_size_sasrec', 'N/A')}")
    logger.info(f"  batch_size_kge   : {config['training'].get('batch_size_kge', 'N/A')}")
    logger.info(f"  num_neg_b        : {config['training'].get('num_neg_b', 'N/A')}")
    logger.info(f"  margin_loss      : {config['training'].get('margin_loss', 'N/A')}")
    logger.info(f"  hidden_dropout   : {config['model'].get('hidden_dropout', 'N/A')}")
    logger.info(f"  attn_dropout     : {config['model'].get('attn_dropout', 'N/A')}")

    logger.info("\n[ 2 step]  Unified ID Space...")
    space = build_unified_id_space(
        kg_train_path=config['paths']['kg_train'],
        kg_valid_path=config['paths']['kg_valid'],
        kg_test_path=config['paths']['kg_test'],
        recbole_train_inter=config['paths']['recbole_train_inter'],
        recbole_valid_inter=config['paths']['recbole_valid_inter'],
        recbole_test_inter=config['paths']['recbole_test_inter'],
        recbole_dataset_name=config['paths']['recbole_dataset_name'],
        recbole_data_path=config['paths']['recbole_data_path'],
        verbose=True,
    )
    logger.info(f"  Total entities (unified): {space.num_total_entities:,}")
    logger.info(f"  Total items (unified): {space.num_unified_items:,}")

    rel_mapping = {
        "bought": 0,
        "compatible_with": 1,
        "compatible_with_model": 2,
        "instance_of": 3,
        "owns": 4,
    }


    logger.info("\n[step 3] loading the dataset RecBole...")
    recbole_cfg = RecBoleConfig(model='SASRec', dataset='b2b_data',
                                config_dict=config['recbole_config'])

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    recbole_dataset = create_dataset(recbole_cfg)
    train_data, valid_data, test_data = data_preparation(recbole_cfg, recbole_dataset)

    num_recbole_items = recbole_dataset.item_num
    logger.info(f"  RecBole items (incl. padding): {num_recbole_items}")

    logger.info("\n[step 4] building the translation tensor RecBole -> Unified...")
    PAD_UNIFIED_ID = space.num_total_entities
    recbole_to_unified_tensor = torch.full((num_recbole_items,), PAD_UNIFIED_ID, dtype=torch.long)
    for recbole_id in range(num_recbole_items):
        uid = space.recbole_to_unified(recbole_id)
        if uid is not None:
            recbole_to_unified_tensor[recbole_id] = uid

    logger.info(f"  tensor built: shape={recbole_to_unified_tensor.shape}, "
                f"nonzero={(recbole_to_unified_tensor != PAD_UNIFIED_ID).sum().item()}")



    logger.info("\n[step 5] building DataLoader Task A (KG, filtered for variant)...")
    kg_train_loader = build_kg_dataloader(
        kg_file_path=config['paths']['kg_train_filtered'],
        entity_to_unified=space.entity_to_unified_id,
        relation_to_id=rel_mapping,
        batch_size=config['training']['batch_size_kge'],
        shuffle=True,
    )

    logger.info("\n[step 6] building DataLoader Task B (Rec)...")
    sasrec_train_loader = build_rec_dataloader(
        recbole_dataloader=train_data,
        recbole_to_unified_tensor=recbole_to_unified_tensor,
        batch_size=config['training']['batch_size_sasrec'],
        shuffle=True,
    )
    logger.info(f"Training sequences: {len(sasrec_train_loader.dataset)}")
    logger.info(f"RecBole total users: {recbole_dataset.user_num}")



    logger.info("\n[step 7] instantiation of JointAlternateModel...")
    model = JointAlternateModel(
        num_entities=space.num_total_entities + 1,
        num_relations=len(rel_mapping),
        num_items=num_recbole_items,
        kge_dim=config['model']['kge_dim'],
        n_layers=config['model'].get('n_layers', 4),
        n_heads=config['model'].get('n_heads', 2),
        hidden_dropout=config['model'].get('hidden_dropout', 0.4),
        attn_dropout=config['model'].get('attn_dropout', 0.3),
        padding_idx=PAD_UNIFIED_ID
    ).to(device)



    logger.info("\n[step 8] Warm Start ")

    pykeen_state_dict = torch.load(config['paths']['pykeen_best_model'],
                                   map_location='cpu', weights_only=False)
    pykeen_entity_weights = pykeen_state_dict['entity_representations.0._embeddings.weight']
    pykeen_relation_weights = pykeen_state_dict['relation_representations.0._embeddings.weight']

    with torch.no_grad():
        n_kg = pykeen_entity_weights.shape[0]
        model.shared_embedding.embedding.weight[:n_kg] = pykeen_entity_weights
    logger.info(f"  loaded {n_kg} entities into the SharedEmbedding (variant: {variant})")

    model.load_pretrained_kge(
        pykeen_relations=pykeen_relation_weights,
        rel_mapping_pykeen=rel_mapping,
        rel_mapping_ours=rel_mapping,
    )

    sasrec_checkpoint = torch.load(config['paths']['sasrec_checkpoint'],
                                   map_location='cpu', weights_only=False)
    if 'state_dict' in sasrec_checkpoint:
        sasrec_state_dict = sasrec_checkpoint['state_dict']
    else:
        sasrec_state_dict = sasrec_checkpoint
    model.load_pretrained_sasrec(sasrec_state_dict)

    logger.info("  Warm Start completato.")
    shared_first_rows = model.shared_embedding.embedding.weight[:100].cpu()
    sasrec_item_emb = sasrec_state_dict['item_embedding.weight']
    logger.info(f"norm sharedemb first 100 rows: {shared_first_rows.norm(dim=1).mean():.4f}")
    logger.info(f"norm sasrec item_embedding first 100 rows: {sasrec_item_emb[:100].norm(dim=1).mean():.4f}")

    logger.info(f"\n[step 8] starting alternating training ({config['training']['scheduling']})...")
    trainer = AlternateTrainer(
        model=model,
        loader_a=kg_train_loader,
        loader_b=sasrec_train_loader,
        num_entities=space.num_total_entities,
        num_items=num_recbole_items,
        learning_rate=config['training']['learning_rate'],
        margin_a=config['training']['margin_loss'],
        scheduling=config['training']['scheduling'],
        device=device,
        num_neg_b=config['training'].get('num_neg_b', 1),
    )

    ckpt_latest_path = os.path.join(out_folder, 'checkpoint_latest.pth')
    start_epoch = 1
    if os.path.exists(ckpt_latest_path):
        try:
            logger.info(f"Found checkpoint {ckpt_latest_path}, loading to resume...")
            ckpt = torch.load(ckpt_latest_path, map_location=device)
            model.load_state_dict(ckpt['model_state'])
            if 'optimizer_state' in ckpt and hasattr(trainer, 'optimizer'):
                try:
                    trainer.optimizer.load_state_dict(ckpt['optimizer_state'])
                except Exception:
                    logger.warning("Could not load optimizer state (incompatible). Continuing without optimizer restore.")
            try:
                _random.setstate(ckpt['random_state'])
                np.random.set_state(ckpt['numpy_state'])
                torch.set_rng_state(ckpt['torch_rng'])
                if device.type == 'cuda' and 'cuda_rngs' in ckpt:
                    for i, rng in enumerate(ckpt['cuda_rngs']):
                        try:
                            torch.cuda.set_rng_state(rng, i)
                        except Exception:
                            pass
            except Exception:
                logger.warning('Could not restore RNG states from checkpoint.')

            training_history = ckpt.get('training_history', [])
            start_epoch = ckpt.get('epoch', 0) + 1
            logger.info(f"Resuming from epoch {start_epoch} (loaded checkpoint epoch {ckpt.get('epoch', 0)})")
        except Exception as e:
            logger.warning(f"Failed loading checkpoint: {e}. Starting from scratch.")
    else:
        training_history = []




    logger.info("\n[Sanity Check] evaluating a epoch 0 on  Warm-Started pth (Baseline to overcome)...")
    val_metrics_zero = evaluate_task_b(model, valid_data, recbole_to_unified_tensor)
    r20_zero = val_metrics_zero.get('Recall@20', 0.0)
    ndcg20_zero = val_metrics_zero.get('NDCG@20', 0.0)
    logger.info(f"  --> BASELINE (epoch 0) | R@20: {r20_zero:.4f} | NDCG@20: {ndcg20_zero:.4f}")


    best_r20 = r20_zero
    best_ndcg = ndcg20_zero
    best_epoch = 0
    best_r20_test = -1.0
    best_epoch_test = 0
    num_epochs = config['training']['epochs']
    eval_step = config['training']['eval_step']

    for epoch in range(start_epoch, num_epochs + 1):
        stats = trainer.train_one_epoch()

        history_entry = {
            "epoch": epoch,
            "loss_A": stats['loss_a_mean'],
            "loss_B": stats['loss_b_mean'],
            "batches_A": stats['batches_a'],
            "batches_B": stats['batches_b'],
            "time_sec": stats['time_sec'],
        }
        training_history.append(history_entry)

        logger.info(
            f"Epoch {epoch}/{num_epochs} | "
            f"LossA: {stats['loss_a_mean']:.4f} | "
            f"LossB: {stats['loss_b_mean']:.4f} | "
            f"batches A/B: {stats['batches_a']}/{stats['batches_b']} | "
            f"time: {stats['time_sec']:.1f}s"
        )

        if epoch % eval_step == 0:
            logger.info("  [Validation] evaluating on validation set...")
            val_metrics = evaluate_task_b(model, valid_data, recbole_to_unified_tensor)
            r20 = val_metrics.get('Recall@20', 0.0)
            ndcg20 = val_metrics.get('NDCG@20', 0.0)
            logger.info(f"  [Validation] R@20={r20:.4f}  NDCG@20={ndcg20:.4f}")

            history_entry['val_recall_20'] = r20
            history_entry['val_ndcg_20'] = ndcg20

            if r20 > best_r20:
                best_r20 = r20
                best_ndcg = ndcg20
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(out_folder, "best_model.pt"))
                logger.info(f"  [Validation] New best model saved (epoch {epoch})")

            logger.info("  [Test] evaluating on test set...")
            test_metrics_epoch = evaluate_task_b(model, test_data, recbole_to_unified_tensor)
            r20_test_epoch = test_metrics_epoch.get('Recall@20', 0.0)
            ndcg20_test_epoch = test_metrics_epoch.get('NDCG@20', 0.0)
            logger.info(f"  [Test] R@20={r20_test_epoch:.4f}  NDCG@20={ndcg20_test_epoch:.4f}")

            history_entry['test_recall_20'] = r20_test_epoch
            history_entry['test_ndcg_20'] = ndcg20_test_epoch

            if r20_test_epoch > best_r20_test:
                best_r20_test = r20_test_epoch
                best_epoch_test = epoch
                torch.save(model.state_dict(), os.path.join(out_folder, "best_model_on_test.pt"))
                logger.info(f"  [Test] New best model (on test) saved (epoch {epoch})")

        with open(os.path.join(out_folder, "training_history.json"), "w") as f:
            json.dump(training_history, f, indent=4)

        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': trainer.optimizer.state_dict() if hasattr(trainer, 'optimizer') else None,
            'training_history': training_history,
            'random_state': _random.getstate(),
            'numpy_state': np.random.get_state(),
            'torch_rng': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            try:
                ckpt['cuda_rngs'] = [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
            except Exception:
                ckpt['cuda_rngs'] = None

        epoch_ckpt_path = os.path.join(out_folder, f'checkpoint_epoch_{epoch}.pth')
        try:
            torch.save(ckpt, epoch_ckpt_path)
            torch.save(ckpt, os.path.join(out_folder, 'checkpoint_latest.pth'))
            logger.info(f"Saved epoch checkpoint to {epoch_ckpt_path}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint for epoch {epoch}: {e}")

    torch.save(model.state_dict(), os.path.join(out_folder, "last_model.pt"))

    logger.info("\n[step 10] final evaluation on test set using best model (on validation)...")

    if os.path.exists(os.path.join(out_folder, "best_model.pt")):
        model.load_state_dict(torch.load(os.path.join(out_folder, "best_model.pt"),
                                         map_location=device))
        logger.info(f"  loaded best model (epoch {best_epoch})")
    else:
        logger.warning("  best_model.pt not found — using last_model.pt")

    test_metrics = evaluate_task_b(model, test_data, recbole_to_unified_tensor)
    r20_test = test_metrics.get('Recall@20', 0.0)
    ndcg20_test = test_metrics.get('NDCG@20', 0.0)

    logger.info("=" * 70)
    logger.info(f"FINAL RESULTS ON TEST SET -- variant: {variant}")
    logger.info("=" * 70)
    logger.info(f"  R@20:    {r20_test:.4f}")
    logger.info(f"  NDCG@20: {ndcg20_test:.4f}")
    logger.info(f"  Best epoch: {best_epoch}")
    logger.info(f"  Best epoch (test): {best_epoch_test}")

    final_results = {
        "variant": variant,
        "kg_train_used": config['paths']['kg_train_filtered'],
        "pykeen_best_model_used": config['paths']['pykeen_best_model'],
        "R@20": r20_test,
        "NDCG@20": ndcg20_test,
        "best_epoch": best_epoch,
        "best_val_R@20": best_r20,
        "best_val_NDCG@20": best_ndcg,
        "best_epoch_test": best_epoch_test,
        "scheduling": config['training']['scheduling'],
        "epochs_run": num_epochs,
        "hyperparameters": {
            "learning_rate": config['training'].get('learning_rate'),
            "batch_size_sasrec": config['training'].get('batch_size_sasrec'),
            "batch_size_kge": config['training'].get('batch_size_kge'),
            "num_neg_b": config['training'].get('num_neg_b'),
            "margin_loss": config['training'].get('margin_loss'),
            "hidden_dropout": config['model'].get('hidden_dropout'),
            "attn_dropout": config['model'].get('attn_dropout'),
        },
    }

    with open(os.path.join(out_folder, "best_metrics.json"), "w") as f:
        json.dump(final_results, f, indent=4)

    logger.info("\nTraining completed.")
    logger.info(f"Final results saved in {out_folder}")


if __name__ == '__main__':
    main()
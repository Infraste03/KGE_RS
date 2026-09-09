"""
run_step4_ablation_fashion_v3.py
==================================

Step 4 orchestrator - Fashion v3 Alternate Learning, ABLATION VARIANTS.

Selected variants:

  - loo_no_compatible_with   (5rel) - negative-control configuration
  - mix3_keep_cat_brand_cw   (5rel) - best isolated TransE MRR, above full KG
  - targeted_pair_cw_brand   (5rel) - minimal two-relation configuration
  - loo_no_belongs_to        (3rel) - direct comparison across relation families

Step 4 hyperparameters
(learning_rate, batch_size_*, margin_loss, dropout*)
are FIXED across all variants and are taken from the winning Step 4 HPO
trial on the full 3-relation configuration.

They are never re-optimized for individual ablation variants. This isolates
the effect of the relation subset and follows the same principle used for
the isolated TransE ablation study.

Differences compared with run_step4_fashion_v3.py:

  1. --variant replaces --relations and selects the ablation configuration.

  2. Filtered KG files and TransE checkpoints are resolved from:

       KG_RS/ablation_study/fashionv3/

     outside fashion_generalization/.

  3. Overwrite protection is enabled through best_metrics.json and --force.

  4. num_relations is ALWAYS 5 because all variants use the same fixed
     relation vocabulary, following the same principle used by the full
     v3 setup.

Usage:

  python alternate_learning/variants/v3/run_step4_ablation_fashion_v3.py \
      --config alternate_learning/configs/step4_config_fashion_v3.yaml \
      --variant loo_no_compatible_with

  python alternate_learning/variants/v3/run_step4_ablation_fashion_v3.py \
      --config alternate_learning/configs/step4_config_fashion_v3.yaml \
      --variant mix3_keep_cat_brand_cw \
      --force
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
import pandas as pd


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# This script is stored under:
#   alternate_learning/variants/v3/
#
# Main alternate-learning modules are stored under:
#   alternate_learning/
ALTERNATE_DIR = os.path.abspath(
    os.path.join(THIS_DIR, "..", "..")
)

MODELS_DIR = os.path.join(
    ALTERNATE_DIR,
    "models"
)

# Fashion project root:
#   fashion_generalization/
BASE_DIR = os.path.abspath(
    os.path.join(ALTERNATE_DIR, "..")
)

# Repository root:
#   KG_RS/
KG_RS_ROOT = os.path.abspath(
    os.path.join(BASE_DIR, "..")
)

if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

if ALTERNATE_DIR not in sys.path:
    sys.path.insert(0, ALTERNATE_DIR)

if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)


from unified_id_space_fashion_v3 import build_unified_id_space_fashion
from load_kg_fashion_v3 import load_kg_fashion
from joint_model_fashion import JointAlternateModelFashion
from data_loaders_fashion_v3 import (
    build_kg_dataloader_fashion,
    build_rec_dataloader_fashion,
)
from alternate_loop_fashion_ce import AlternateTrainerFashion
from eval_utils_fashion_v3 import (
    evaluate_task_a_fashion,
    evaluate_task_b_fashion,
    reset_eval_debug,
)

import eval_utils_fashion_v3 as _eu
_eu._EVAL_TASK_B_DEBUG_PRINTED = True


from recbole.config import Config as RecBoleConfig
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed
from pykeen.triples import TriplesFactory


# Fashion v3 constants.
# A single unified vocabulary is shared across
# 3rel, 5rel, and ablation configurations.
UNIFIED_PADDING = 81_259
ITEM_ID_START   = 9_317
ITEM_ID_END     = 81_251
NUM_KG_ENT      = 81_259

# Always 5 because the same fixed relation vocabulary
# is preserved across all variants.
NUM_RELATIONS = 5


DATA_DIR_V3 = os.path.join(
    BASE_DIR,
    "data",
    "processed_v3"
)

KG_VALID = os.path.join(
    DATA_DIR_V3,
    "taskA_valid.tsv"
)

KG_TEST = os.path.join(
    DATA_DIR_V3,
    "taskA_test.tsv"
)

# The complete KG is always used to construct the fixed vocabulary.
KG_TRAIN_FULL = os.path.join(
    DATA_DIR_V3,
    "kg_train.tsv"
)

ENTITY2ID_PATH = os.path.join(
    DATA_DIR_V3,
    "entity2id.tsv"
)

RELATION2ID_PATH = os.path.join(
    DATA_DIR_V3,
    "relation2id.tsv"
)


# Ablation study directory located at the KG_RS repository root.
ABLATION_STUDY_BASE = os.path.join(
    KG_RS_ROOT,
    "ablation_study",
    "fashionv3"
)


# Variant -> relation family.
# This determines which TransE ablation result directory is used.
VARIANT_FAMILY = {
    "loo_no_compatible_with": 5,
    "mix3_keep_cat_brand_cw": 5,
    "targeted_pair_cw_brand": 5,
    "loo_no_belongs_to":      3,
}


def resolve_ablation_paths(
    variant_name: str,
    logger
) -> dict:

    if variant_name not in VARIANT_FAMILY:
        raise ValueError(
            f"Invalid variant '{variant_name}'. "
            f"Available variants: {list(VARIANT_FAMILY.keys())}"
        )

    family = VARIANT_FAMILY[
        variant_name
    ]

    kg_train_filtered = os.path.join(
        ABLATION_STUDY_BASE,
        f"ablation_{family}rel",
        f"kg_train_{variant_name}.tsv"
    )

    transe_checkpoint = os.path.join(
        ABLATION_STUDY_BASE,
        f"results_{family}rel",
        variant_name,
        f"best_model_{variant_name}.pt"
    )

    if not os.path.exists(
        kg_train_filtered
    ):
        raise FileNotFoundError(
            f"Filtered KG not found: "
            f"{kg_train_filtered}"
        )

    if not os.path.exists(
        transe_checkpoint
    ):
        raise FileNotFoundError(
            f"TransE warm-start checkpoint not found: "
            f"{transe_checkpoint}"
        )

    logger.info(
        f"  Variant '{variant_name}' -> "
        f"{family}-relation family"
    )

    logger.info(
        f"  Filtered KG : "
        f"{kg_train_filtered}"
    )

    logger.info(
        f"  TransE ckpt : "
        f"{transe_checkpoint}"
    )

    return {
        "family": family,
        "kg_train_filtered": kg_train_filtered,
        "transe_checkpoint": transe_checkpoint,
    }


def load_pykeen_vocab(logger):
    """
    Load the exact PyKEEN vocabulary used during Fashion v3 TransE HPO.

    The same vocabulary is shared across all ablation variants.
    """
    e2id_df = pd.read_csv(
        ENTITY2ID_PATH,
        sep='\t'
    )

    r2id_df = pd.read_csv(
        RELATION2ID_PATH,
        sep='\t'
    )

    pykeen_e2id = dict(
        zip(
            e2id_df["entity"],
            e2id_df["id"]
        )
    )

    pykeen_r2id = dict(
        zip(
            r2id_df["relation"],
            r2id_df["id"]
        )
    )

    logger.info(
        f"  PyKEEN vocabulary: "
        f"{len(pykeen_e2id):,} entities, "
        f"{len(pykeen_r2id)} relations"
    )

    return pykeen_e2id, pykeen_r2id


# ============================================================================ #
# Configuration and logging
# ============================================================================ #

def load_config():

    parser = argparse.ArgumentParser(
        description=(
            "Step 4 Fashion v3 - "
            "Alternate Learning (ABLATION)"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True
    )

    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=list(
            VARIANT_FAMILY.keys()
        )
    )

    parser.add_argument(
        "--scheduling",
        type=str,
        choices=[
            "one_to_one",
            "epoch",
            "adaptive"
        ]
    )

    parser.add_argument(
        "--epochs",
        type=int
    )

    parser.add_argument(
        "--run-tag",
        type=str,
        default=""
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Run again even if "
            "best_metrics.json already exists"
        )
    )

    args = parser.parse_args()

    with open(
        args.config,
        'r'
    ) as f:
        config = yaml.safe_load(f)

    if args.scheduling:
        config[
            'training'
        ][
            'scheduling'
        ] = args.scheduling

    if args.epochs:
        config[
            'training'
        ][
            'epochs'
        ] = args.epochs

    if args.run_tag:
        config[
            'training'
        ][
            'run_tag'
        ] = args.run_tag

    if args.seed is not None:
        config[
            'recbole_config'
        ][
            'seed'
        ] = args.seed

    config['_variant'] = args.variant
    config['_force'] = args.force

    return config


def setup_logging(output_dir: str):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    log_file = os.path.join(
        output_dir,
        'run.log'
    )

    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%d-%b %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
    )

    return logging.getLogger(__name__)


def build_type_index(kg: dict) -> dict:

    type_to_ids = {
        'item': [],
        'category': [],
        'brand': [],
        'pricetier': [],
        'poptier': []
    }

    for label, eid in kg[
        'entity_to_id'
    ].items():

        prefix = label.split(
            '_'
        )[0]

        if prefix in type_to_ids:
            type_to_ids[
                prefix
            ].append(eid)

    return {
        k: torch.LongTensor(v)
        for k, v in type_to_ids.items()
    }


# ============================================================================ #
# Main
# ============================================================================ #

def main():

    config = load_config()

    variant = config[
        '_variant'
    ]

    run_tag = config[
        'training'
    ].get(
        'run_tag',
        ''
    )

    sched = config[
        'training'
    ][
        'scheduling'
    ]


    # Keep Step 4 v3 outputs under alternate_learning/,
    # consistently with run_step4_fashion_v3.py and the HPO wrapper.
    out_dir = os.path.join(
        ALTERNATE_DIR,
        "hpc_step4_results_v3",
        f"step4_ablation_{variant}_{sched}_ce{run_tag}"
    )


    best_metrics_path = os.path.join(
        out_dir,
        "best_metrics.json"
    )


    if (
        os.path.exists(best_metrics_path)
        and not config['_force']
    ):

        print(
            f"[SKIP] Variant '{variant}' ({sched}) "
            f"already completed: {best_metrics_path}"
        )

        print(
            "       Use --force to run it again."
        )

        sys.exit(0)


    logger = setup_logging(
        out_dir
    )


    logger.info("=" * 70)

    logger.info(
        f"STEP 4 FASHION v3 - "
        f"ALTERNATE LEARNING (ABLATION) - "
        f"{variant}"
    )

    logger.info("=" * 70)

    logger.info(
        f"Output dir : {out_dir}"
    )

    logger.info(
        f"Scheduling : {sched}"
    )

    logger.info(
        f"Epochs     : "
        f"{config['training']['epochs']}"
    )


    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    logger.info(
        f"Device     : {device}"
    )


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


    ablation_paths = resolve_ablation_paths(
        variant,
        logger
    )


    # ===================================================================== #
    # Unified ID space
    #
    # ALWAYS built from the complete KG so that every ablation variant
    # uses the same fixed vocabulary as the full 3rel/5rel configurations.
    # ===================================================================== #

    logger.info(
        "\n[2/10] Building Fashion v3 Unified ID Space..."
    )


    INTER_PATH = os.path.join(
        BASE_DIR,
        config[
            'paths'
        ][
            'recbole_inter_path'
        ]
    )

    RECBOLE_DIR = os.path.join(
        BASE_DIR,
        config[
            'paths'
        ][
            'recbole_data_path'
        ]
    )

    # SASRec checkpoint selected in the Fashion v3 YAML configuration.
    SASREC_PATH = os.path.join(
        BASE_DIR,
        config[
            'paths'
        ][
            'sasrec_checkpoint'
        ]
    )


    space = build_unified_id_space_fashion(
        kg_train_path=KG_TRAIN_FULL,
        kg_valid_path=KG_VALID,
        kg_test_path=KG_TEST,
        recbole_inter_path=INTER_PATH,
        recbole_data_path=RECBOLE_DIR,
        verbose=True,
    )


    logger.info(
        f"  Total KG entities: "
        f"{space.num_total_entities:,}"
    )


    assert space.num_total_entities == NUM_KG_ENT, \
        (
            f"Expected {NUM_KG_ENT} entities, "
            f"found {space.num_total_entities}!"
        )


    kg = load_kg_fashion(
        KG_TRAIN_FULL,
        KG_VALID,
        KG_TEST,
        verbose=False
    )


    assert kg['num_relations'] == NUM_RELATIONS, \
        (
            f"Expected {NUM_RELATIONS} relations, "
            f"found {kg['num_relations']}"
        )


    type_index = build_type_index(
        kg
    )

    n_items = len(
        type_index[
            'item'
        ]
    )


    assert n_items == 71_935, \
        (
            f"Expected 71935 items, "
            f"found {n_items} - "
            f"check ITEM_ID_START/END!"
        )


    # ===================================================================== #
    # RecBole dataset
    #
    # Identical across all variants because Task B never changes.
    # ===================================================================== #

    logger.info(
        "\n[3/10] Loading Fashion v3 RecBole dataset..."
    )


    recbole_cfg_dict = dict(
        config[
            'recbole_config'
        ]
    )

    recbole_cfg_dict[
        'data_path'
    ] = RECBOLE_DIR


    recbole_cfg = RecBoleConfig(
        model='SASRec',
        dataset='fashion_v3',
        config_dict=recbole_cfg_dict
    )


    init_seed(
        recbole_cfg[
            'seed'
        ],
        recbole_cfg[
            'reproducibility'
        ]
    )


    recbole_dataset = create_dataset(
        recbole_cfg
    )


    train_data_recbole, valid_data, test_data = data_preparation(
        recbole_cfg,
        recbole_dataset
    )


    num_items_recbole = recbole_dataset.item_num


    logger.info(
        f"  item_num (including PAD): "
        f"{num_items_recbole:,}"
    )


    # ===================================================================== #
    # RecBole -> Unified tensor
    # ===================================================================== #

    logger.info(
        "\n[4/10] Building r2u tensor..."
    )


    r2u = torch.full(
        (num_items_recbole,),
        UNIFIED_PADDING,
        dtype=torch.long
    )


    for recbole_id in range(
        num_items_recbole
    ):

        uid = space.recbole_to_unified(
            recbole_id
        )

        if uid is not None:
            r2u[
                recbole_id
            ] = uid


    n_mapped = (
        r2u != UNIFIED_PADDING
    ).sum().item()


    assert n_mapped == num_items_recbole - 1, \
        (
            f"Expected {num_items_recbole - 1} mapped items, "
            f"found {n_mapped}"
        )


    # ===================================================================== #
    # DataLoaders
    #
    # Task A uses the FILTERED KG associated with the selected variant.
    # ===================================================================== #

    logger.info(
        f"\n[5/10] Building DataLoaders "
        f"(variant: {variant})..."
    )


    kg_train_loader = build_kg_dataloader_fashion(
        kg_file_path=ablation_paths[
            'kg_train_filtered'
        ],
        entity_to_unified=kg[
            'entity_to_id'
        ],
        relation_to_id=kg[
            'relation_to_id'
        ],
        batch_size=config[
            'training'
        ][
            'batch_size_kge'
        ],
        shuffle=True,
    )


    logger.info(
        f"  KG loader ({variant}): "
        f"{len(kg_train_loader.dataset):,} triples, "
        f"{len(kg_train_loader)} batches"
    )


    sasrec_train_loader = build_rec_dataloader_fashion(
        recbole_train_dataloader=train_data_recbole,
        recbole_to_unified_tensor=r2u,
        batch_size=config[
            'training'
        ][
            'batch_size_sasrec'
        ],
        shuffle=True,
    )


    logger.info(
        f"  Rec loader: "
        f"{len(sasrec_train_loader.dataset):,} sequences, "
        f"{len(sasrec_train_loader)} batches"
    )


    # ===================================================================== #
    # Model
    # ===================================================================== #

    logger.info(
        "\n[6/10] Instantiating JointAlternateModelFashion..."
    )


    model = JointAlternateModelFashion(
        num_entities=NUM_KG_ENT + 1,
        num_relations=NUM_RELATIONS,
        kge_dim=config[
            'model'
        ][
            'kge_dim'
        ],
        max_seq_length=recbole_cfg_dict.get(
            'MAX_ITEM_LIST_LENGTH',
            50
        ),
        n_layers=config[
            'model'
        ][
            'n_layers'
        ],
        n_heads=config[
            'model'
        ][
            'n_heads'
        ],
        inner_size=config[
            'model'
        ][
            'inner_size'
        ],
        hidden_dropout=config[
            'model'
        ][
            'hidden_dropout'
        ],
        attn_dropout=config[
            'model'
        ][
            'attn_dropout'
        ],
        unified_padding_idx=UNIFIED_PADDING,
    ).to(device)


    # ===================================================================== #
    # Warm start
    #
    # Variant-specific TransE + fixed SASRec checkpoint.
    # ===================================================================== #

    logger.info(
        "\n[7/10] Warm starting from pretrained checkpoints..."
    )


    transe_path = ablation_paths[
        'transe_checkpoint'
    ]


    transe_state = torch.load(
        transe_path,
        map_location='cpu',
        weights_only=False
    )


    entity_weights = transe_state[
        'entity_representations.0._embeddings.weight'
    ]

    relation_weights = transe_state[
        'relation_representations.0._embeddings.weight'
    ]


    pykeen_e2id, pykeen_r2id = load_pykeen_vocab(
        logger
    )


    model.load_pretrained_kge(
        pykeen_entity_weights=entity_weights,
        entity_mapping_pykeen=pykeen_e2id,
        entity_mapping_ours=kg[
            'entity_to_id'
        ],
        pykeen_relation_weights=relation_weights,
        rel_mapping_pykeen=pykeen_r2id,
        rel_mapping_ours=kg[
            'relation_to_id'
        ],
    )


    sasrec_ckpt = torch.load(
        SASREC_PATH,
        map_location='cpu',
        weights_only=False
    )


    sasrec_state = sasrec_ckpt.get(
        'state_dict',
        sasrec_ckpt
    )


    model.load_pretrained_sasrec(
        sasrec_state
    )


    logger.info(
        "  Warm start completed."
    )


    # ===================================================================== #
    # Trainer
    #
    # Hyperparameters are FIXED and come from the winning Step 4 HPO
    # trial on the full 3-relation Fashion v3 configuration.
    # ===================================================================== #

    logger.info(
        f"\n[8/10] Instantiating "
        f"AlternateTrainerFashion ({sched})..."
    )


    trainer = AlternateTrainerFashion(
        model=model,
        loader_a=kg_train_loader,
        loader_b=sasrec_train_loader,
        num_entities=NUM_KG_ENT,
        learning_rate=config[
            'training'
        ][
            'learning_rate'
        ],
        margin_a=config[
            'training'
        ][
            'margin_loss'
        ],
        scheduling=sched,
        device=str(device),
        log_every=200,
        item_id_start=ITEM_ID_START,
        item_id_end=ITEM_ID_END,
    )


    assert trainer.loss_b_fn.__class__.__name__ == 'SASRecCELoss', \
        (
            f"Expected SASRecCELoss, got "
            f"{trainer.loss_b_fn.__class__.__name__}"
        )


    logger.info(
        f"  Task B loss confirmed: "
        f"{trainer.loss_b_fn.__class__.__name__}"
    )


    start_epoch = 1
    training_history = []


    ckpt_latest_path = os.path.join(
        out_dir,
        'checkpoint_latest.pth'
    )


    if os.path.exists(
        ckpt_latest_path
    ):

        ckpt = torch.load(
            ckpt_latest_path,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(
            ckpt[
                'model_state'
            ]
        )

        try:
            trainer.optimizer.load_state_dict(
                ckpt[
                    'optimizer_state'
                ]
            )

        except Exception:
            logger.warning(
                "  Optimizer state is incompatible and will be ignored."
            )

        training_history = ckpt.get(
            'training_history',
            []
        )

        start_epoch = ckpt.get(
            'epoch',
            0
        ) + 1

        logger.info(
            f"  Resuming from epoch "
            f"{start_epoch}"
        )


    logger.info(
        "\n[Sanity check] "
        "Evaluating warm start (epoch 0)..."
    )


    reset_eval_debug()


    val0 = evaluate_task_b_fashion(
        model,
        valid_data,
        r2u
    )


    test0 = evaluate_task_b_fashion(
        model,
        test_data,
        r2u
    )


    logger.info(
        f"  Epoch 0 | "
        f"Valid: R@20={val0['Recall@20']:.4f} "
        f"NDCG@20={val0['NDCG@20']:.4f} | "
        f"Test: R@20={test0['Recall@20']:.4f} "
        f"NDCG@20={test0['NDCG@20']:.4f}"
    )


    # ===================================================================== #
    # Training loop
    # ===================================================================== #

    logger.info(
        f"\n[9/10] Alternate training - "
        f"{config['training']['epochs']} epochs..."
    )


    best_r20_val, best_ndcg_val, best_epoch_val = (
        -1.0,
        -1.0,
        0
    )

    best_r20_test, best_epoch_test = (
        -1.0,
        0
    )


    num_epochs = config[
        'training'
    ][
        'epochs'
    ]


    eval_step = config[
        'training'
    ].get(
        'eval_step',
        1
    )


    for epoch in range(
        start_epoch,
        num_epochs + 1
    ):

        stats = trainer.train_one_epoch()


        entry = {
            "epoch":
                epoch,

            "loss_A":
                stats[
                    'loss_a_mean'
                ],

            "loss_B":
                stats[
                    'loss_b_mean'
                ],

            "batches_A":
                stats[
                    'batches_a'
                ],

            "batches_B":
                stats[
                    'batches_b'
                ],

            "time_sec":
                stats[
                    'time_sec'
                ]
        }


        training_history.append(
            entry
        )


        logger.info(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"LossA={stats['loss_a_mean']:.4f} | "
            f"LossB={stats['loss_b_mean']:.4f} | "
            f"time={stats['time_sec']:.1f}s"
        )


        if epoch % eval_step == 0:


            reset_eval_debug()


            val_m = evaluate_task_b_fashion(
                model,
                valid_data,
                r2u
            )


            r20_v, ndcg_v = (
                val_m[
                    'Recall@20'
                ],
                val_m[
                    'NDCG@20'
                ]
            )


            entry[
                'val_recall_20'
            ], entry[
                'val_ndcg_20'
            ] = r20_v, ndcg_v


            reset_eval_debug()


            test_m = evaluate_task_b_fashion(
                model,
                test_data,
                r2u
            )


            r20_t, ndcg_t = (
                test_m[
                    'Recall@20'
                ],
                test_m[
                    'NDCG@20'
                ]
            )


            entry[
                'test_recall_20'
            ], entry[
                'test_ndcg_20'
            ] = r20_t, ndcg_t


            logger.info(
                f"  ==> Valid: "
                f"R@20={r20_v:.4f} "
                f"NDCG@20={ndcg_v:.4f} | "
                f"Test: "
                f"R@20={r20_t:.4f} "
                f"NDCG@20={ndcg_t:.4f}"
            )


            if r20_v > best_r20_val:

                best_r20_val, best_ndcg_val, best_epoch_val = (
                    r20_v,
                    ndcg_v,
                    epoch
                )

                torch.save(
                    model.state_dict(),
                    os.path.join(
                        out_dir,
                        "best_model.pt"
                    )
                )

                logger.info(
                    f"  => NEW BEST VALID | "
                    f"epoch={epoch}"
                )


            if r20_t > best_r20_test:

                best_r20_test, best_epoch_test = (
                    r20_t,
                    epoch
                )

                torch.save(
                    model.state_dict(),
                    os.path.join(
                        out_dir,
                        "best_model_on_test.pt"
                    )
                )


        with open(
            os.path.join(
                out_dir,
                "training_history.json"
            ),
            "w"
        ) as f:

            json.dump(
                training_history,
                f,
                indent=4
            )


        ckpt = {
            'epoch':
                epoch,

            'model_state':
                model.state_dict(),

            'optimizer_state':
                trainer.optimizer.state_dict(),

            'training_history':
                training_history,

            'random_state':
                _random.getstate(),

            'numpy_state':
                np.random.get_state(),

            'torch_rng':
                torch.get_rng_state()
        }


        torch.save(
            ckpt,
            os.path.join(
                out_dir,
                'checkpoint_latest.pth'
            )
        )


    torch.save(
        model.state_dict(),
        os.path.join(
            out_dir,
            "last_model.pt"
        )
    )


    # ===================================================================== #
    # Final evaluation
    # ===================================================================== #

    logger.info(
        "\n[10/10] Final evaluation..."
    )


    best_model_path = os.path.join(
        out_dir,
        "best_model.pt"
    )


    if os.path.exists(
        best_model_path
    ):

        model.load_state_dict(
            torch.load(
                best_model_path,
                map_location=device,
                weights_only=False
            )
        )


    reset_eval_debug()


    final_test = evaluate_task_b_fashion(
        model,
        test_data,
        r2u
    )


    r20_final, ndcg_final = (
        final_test[
            'Recall@20'
        ],
        final_test[
            'NDCG@20'
        ]
    )


    logger.info("=" * 70)

    logger.info(
        f"FINAL RESULTS - "
        f"{variant} - TEST SET"
    )

    logger.info("=" * 70)

    logger.info(
        f"  Recall@20 : "
        f"{r20_final:.4f}"
    )

    logger.info(
        f"  NDCG@20   : "
        f"{ndcg_final:.4f}"
    )


    final_results = {
        "variant":
            variant,

        "family":
            ablation_paths[
                'family'
            ],

        "kg_train_filtered_used":
            ablation_paths[
                'kg_train_filtered'
            ],

        "transe_checkpoint_used":
            transe_path,

        "test_from_best_valid_recall":
            r20_final,

        "test_from_best_valid_ndcg":
            ndcg_final,

        "best_epoch_valid":
            best_epoch_val,

        "best_val_recall_20":
            best_r20_val,

        "best_val_ndcg_20":
            best_ndcg_val,

        "best_epoch_test":
            best_epoch_test,

        "best_test_recall_20":
            best_r20_test,

        "scheduling":
            sched,

        "epochs_run":
            num_epochs,
    }


    with open(
        os.path.join(
            out_dir,
            "best_metrics.json"
        ),
        "w"
    ) as f:

        json.dump(
            final_results,
            f,
            indent=4
        )


    logger.info(
        f"\nTraining completed. "
        f"Results saved in {out_dir}"
    )


if __name__ == '__main__':
    main()
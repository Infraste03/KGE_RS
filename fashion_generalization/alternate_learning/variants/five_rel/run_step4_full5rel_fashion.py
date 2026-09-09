"""
run_step4_full5rel_fashion.py
==============================

Step 4 orchestrator — Fashion Alternate Learning with the COMPLETE
5-RELATION KG.

The KG is stored under processed_v2 and contains:
  - compatible_with
  - belongs_to
  - belongs_to_brand
  - belongs_to_price_tier
  - belongs_to_pop_tier

Differences compared with run_step4_ablation_fashion.py:

  1. There is no --variant logic: this script uses one complete graph,
     not an ablation variant.

  2. NUM_KG_ENT = 182_990 instead of 182_983 because the 5-relation KG
     contains 7 additional shared price/popularity tier entities.

  3. It uses the dedicated 5-relation modules:
       - load_kg_fashion_5rel
       - unified_id_space_fashion_5rel
       - data_loaders_fashion_5rel

     These modules recognize the additional pricetier_ and poptier_
     entity prefixes.

  4. The TransE warm start is loaded from:

       results/hpo_transe_full5rel_v2/best_model.pt

     The checkpoint uses embedding dimension 64.

  5. The SASRec warm start is the same one used by the complete
     3-relation model. Items are identical between processed/ and
     processed_v2/, so the same SASRec checkpoint remains valid.

  6. Task B uses Cross-Entropy through alternate_loop_fashion_ce.

Usage:
  python run_step4_full5rel_fashion.py --config configs/step4_hpc_config_fashion_ce.yaml
  python run_step4_full5rel_fashion.py --config configs/step4_hpc_config_fashion_ce.yaml --scheduling epoch
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

# This script is stored in:
#   alternate_learning/variants/five_rel/
#
# Main alternate-learning modules are stored in:
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

if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

if ALTERNATE_DIR not in sys.path:
    sys.path.insert(0, ALTERNATE_DIR)

if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)


from unified_id_space_fashion_5rel import build_unified_id_space_fashion
from load_kg_fashion_5rel import load_kg_fashion
from joint_model_fashion import JointAlternateModelFashion
from data_loaders_fashion_5rel import (
    build_kg_dataloader_fashion,
    build_rec_dataloader_fashion,
)
from alternate_loop_fashion_ce import AlternateTrainerFashion
from eval_utils_fashion import (
    evaluate_task_a_fashion,
    evaluate_task_b_fashion,
    reset_eval_debug,
)

import eval_utils_fashion as _eu
_eu._EVAL_TASK_B_DEBUG_PRINTED = True


from recbole.config import Config as RecBoleConfig
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed
from pykeen.triples import TriplesFactory


UNIFIED_PADDING = 182_990
ITEM_ID_START   = 17_514
NUM_KG_ENT      = 182_990

BASELINE_RECALL20 = 0.0964
BASELINE_NDCG20   = 0.0889


# Pretrained TransE checkpoint for the complete 5-relation KG
TRANSE_PATH = os.path.join(
    BASE_DIR,
    "results",
    "hpo_transe_full5rel_v2",
    "best_model.pt",
)


def load_config():

    parser = argparse.ArgumentParser(
        description=(
            "Step 4 Fashion — Alternate Learning "
            "(complete 5-relation KG)"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(
            ALTERNATE_DIR,
            "configs",
            "step4_hpc_config_fashion_ce.yaml",
        ),
    )

    parser.add_argument(
        "--scheduling",
        type=str,
        choices=[
            "one_to_one",
            "epoch",
            "adaptive",
        ],
    )

    parser.add_argument(
        "--epochs",
        type=int,
    )

    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Run again even if best_metrics.json already exists",
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
            logging.StreamHandler(),
        ]
    )

    return logging.getLogger(__name__)


def build_type_index(kg: dict) -> dict:

    # Five entity types:
    # item, category, brand, pricetier, and poptier.
    #
    # Types are detected dynamically from the prefixes present
    # in the KG vocabulary rather than being hardcoded.
    type_to_ids = {}

    for label, eid in kg[
        'entity_to_id'
    ].items():

        prefix = label.split(
            '_'
        )[0]

        type_to_ids.setdefault(
            prefix,
            []
        ).append(eid)

    return {
        k: torch.LongTensor(v)
        for k, v in type_to_ids.items()
    }


def main():

    config = load_config()

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


    if not os.path.exists(
        TRANSE_PATH
    ):
        raise FileNotFoundError(
            f"TransE warm-start weights not found: "
            f"{TRANSE_PATH}"
        )


    # Keep generated experiment outputs under the standard
    # alternate_learning/hpc_step4_results directory even though
    # this runner is stored inside variants/five_rel/.
    out_dir = os.path.join(
        ALTERNATE_DIR,
        "hpc_step4_results",
        f"step4_full5rel_{sched}_ce{run_tag}",
    )


    best_metrics_path = os.path.join(
        out_dir,
        "best_metrics.json",
    )

    if (
        os.path.exists(best_metrics_path)
        and not config['_force']
    ):

        print(
            f"[SKIP] full5rel ({sched}) already completed: "
            f"{best_metrics_path}"
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
        "STEP 4 FASHION — ALTERNATE LEARNING TRAINING "
        "(COMPLETE 5-RELATION KG)"
    )
    logger.info("=" * 70)

    logger.info(
        f"TransE warm start : {TRANSE_PATH}"
    )

    logger.info(
        f"Output dir        : {out_dir}"
    )

    logger.info(
        f"Scheduling        : {sched}"
    )

    logger.info(
        f"Epochs            : "
        f"{config['training']['epochs']}"
    )

    logger.info(
        "Hyperparameters:"
    )

    logger.info(
        f"  lr             = "
        f"{config['training']['learning_rate']}"
    )

    logger.info(
        f"  batch_size_kge = "
        f"{config['training']['batch_size_kge']}"
    )

    logger.info(
        f"  batch_size_rec = "
        f"{config['training']['batch_size_sasrec']}"
    )

    logger.info(
        f"  margin_a       = "
        f"{config['training']['margin_loss']}"
    )

    logger.info(
        f"  hidden_dropout = "
        f"{config['model']['hidden_dropout']}"
    )

    logger.info(
        f"  attn_dropout   = "
        f"{config['model']['attn_dropout']}"
    )


    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    logger.info(
        f"Device            : {device}"
    )


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


    # ===================================================================== #
    # STEP 2: Unified ID space
    #         Complete 5-relation KG from processed_v2
    # ===================================================================== #

    logger.info(
        "\n[2/10] Building Fashion Unified ID Space "
        "(5 relations, processed_v2)..."
    )


    KG_TRAIN = os.path.join(
        BASE_DIR,
        "data",
        "processed_v2",
        "kg_train.tsv",
    )

    KG_VALID = os.path.join(
        BASE_DIR,
        "data",
        "processed_v2",
        "taskA_valid.tsv",
    )

    KG_TEST = os.path.join(
        BASE_DIR,
        "data",
        "processed_v2",
        "taskA_test.tsv",
    )


    INTER_PATH = os.path.join(
        BASE_DIR,
        config[
            'paths'
        ][
            'recbole_inter_path'
        ],
    )

    RECBOLE_DIR = os.path.join(
        BASE_DIR,
        config[
            'paths'
        ][
            'recbole_data_path'
        ],
    )


    # Full SASRec checkpoint.
    # Task B is identical to the 3-relation configuration.
    SASREC_PATH = os.path.join(
        BASE_DIR,
        config[
            'paths'
        ][
            'sasrec_checkpoint'
        ],
    )


    space = build_unified_id_space_fashion(
        kg_train_path=KG_TRAIN,
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
            f"found {space.num_total_entities}"
        )


    kg = load_kg_fashion(
        KG_TRAIN,
        KG_VALID,
        KG_TEST,
        verbose=False,
    )


    type_index = build_type_index(
        kg
    )


    logger.info(
        f"  Items: "
        f"{len(type_index.get('item', [])):,}, "
        f"Categories: "
        f"{len(type_index.get('category', [])):,}, "
        f"Brands: "
        f"{len(type_index.get('brand', [])):,}, "
        f"Pricetier: "
        f"{len(type_index.get('pricetier', [])):,}, "
        f"Poptier: "
        f"{len(type_index.get('poptier', [])):,}"
    )


    # ===================================================================== #
    # STEP 3: RecBole dataset
    #         Identical to the complete 3-relation setup because
    #         Task B does not change
    # ===================================================================== #

    logger.info(
        "\n[3/10] Loading Fashion RecBole dataset..."
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
        dataset='fashion',
        config_dict=recbole_cfg_dict,
    )


    init_seed(
        recbole_cfg[
            'seed'
        ],
        recbole_cfg[
            'reproducibility'
        ],
    )


    recbole_dataset = create_dataset(
        recbole_cfg
    )


    train_data_recbole, valid_data, test_data = data_preparation(
        recbole_cfg,
        recbole_dataset,
    )


    num_items_recbole = recbole_dataset.item_num


    logger.info(
        f"  item_num (including PAD): "
        f"{num_items_recbole:,}"
    )

    logger.info(
        f"  user_num                : "
        f"{recbole_dataset.user_num:,}"
    )


    # ===================================================================== #
    # STEP 4: RecBole -> Unified translation tensor
    # ===================================================================== #

    logger.info(
        "\n[4/10] Building r2u tensor "
        "(RecBole ==> Unified)..."
    )


    r2u = torch.full(
        (num_items_recbole,),
        UNIFIED_PADDING,
        dtype=torch.long,
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


    logger.info(
        f"  r2u shape={r2u.shape}, "
        f"mapped={n_mapped:,}, "
        f"padding entries="
        f"{num_items_recbole - n_mapped}"
    )


    assert n_mapped == num_items_recbole - 1, \
        (
            f"Expected {num_items_recbole - 1} mapped items, "
            f"found {n_mapped}"
        )


    # ===================================================================== #
    # STEP 5: DataLoaders
    #         Task A uses the complete 5-relation training KG
    # ===================================================================== #

    logger.info(
        "\n[5/10] Building DataLoaders..."
    )


    kg_train_loader = build_kg_dataloader_fashion(
        kg_file_path=KG_TRAIN,
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
        f"  KG loader (5rel): "
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
    # STEP 6: Model
    # ===================================================================== #

    logger.info(
        "\n[6/10] Instantiating JointAlternateModelFashion..."
    )


    model = JointAlternateModelFashion(
        num_entities=NUM_KG_ENT + 1,
        num_relations=kg[
            'num_relations'
        ],
        kge_dim=config[
            'model'
        ][
            'kge_dim'
        ],
        max_seq_length=recbole_cfg_dict.get(
            'MAX_ITEM_LIST_LENGTH',
            50,
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
    # STEP 7: Warm start
    #         Complete 5-relation TransE + full 3-relation SASRec
    # ===================================================================== #

    logger.info(
        "\n[7/10] Warm starting from pretrained checkpoints..."
    )


    transe_state = torch.load(
        TRANSE_PATH,
        map_location='cpu',
        weights_only=False,
    )


    entity_weights = transe_state[
        'entity_representations.0._embeddings.weight'
    ]

    relation_weights = transe_state[
        'relation_representations.0._embeddings.weight'
    ]


    train_df = pd.read_csv(
        KG_TRAIN,
        sep='\t',
    )


    train_factory = TriplesFactory.from_labeled_triples(
        train_df[
            [
                'head',
                'relation',
                'tail',
            ]
        ].values
    )


    pykeen_e2id = dict(
        train_factory.entity_to_id
    )

    pykeen_r2id = dict(
        train_factory.relation_to_id
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
        weights_only=False,
    )


    sasrec_state = sasrec_ckpt.get(
        'state_dict',
        sasrec_ckpt,
    )


    model.load_pretrained_sasrec(
        sasrec_state
    )


    logger.info(
        "  Warm start completed."
    )


    logger.info(
        f"  Mean L2 norm of SharedEmbedding "
        f"(first 100 rows): "
        f"{model.shared_embedding.embedding.weight[:100].norm(dim=1).mean():.4f}"
    )


    # ===================================================================== #
    # STEP 8: Trainer (CE)
    # ===================================================================== #

    logger.info(
        f"\n[8/10] Instantiating "
        f"AlternateTrainerFashion CE ({sched})..."
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
    )


    assert trainer.loss_b_fn.__class__.__name__ == 'SASRecCELoss', \
        (
            f"Expected SASRecCELoss, got "
            f"{trainer.loss_b_fn.__class__.__name__} — "
            f"the wrong trainer is being used!"
        )


    logger.info(
        f"  Task B loss confirmed: "
        f"{trainer.loss_b_fn.__class__.__name__}"
    )


    ckpt_latest_path = os.path.join(
        out_dir,
        'checkpoint_latest.pth',
    )

    start_epoch      = 1
    training_history = []


    if os.path.exists(
        ckpt_latest_path
    ):

        try:

            logger.info(
                f"  Checkpoint found: "
                f"{ckpt_latest_path}"
            )


            ckpt = torch.load(
                ckpt_latest_path,
                map_location=device,
                weights_only=False,
            )


            model.load_state_dict(
                ckpt[
                    'model_state'
                ]
            )


            if (
                'optimizer_state' in ckpt
                and ckpt[
                    'optimizer_state'
                ] is not None
            ):

                try:

                    trainer.optimizer.load_state_dict(
                        ckpt[
                            'optimizer_state'
                        ]
                    )

                except Exception:

                    logger.warning(
                        "  Optimizer state is incompatible "
                        "and will be ignored."
                    )


            try:

                _random.setstate(
                    ckpt[
                        'random_state'
                    ]
                )

                np.random.set_state(
                    ckpt[
                        'numpy_state'
                    ]
                )

                torch.set_rng_state(
                    ckpt[
                        'torch_rng'
                    ]
                )


                if (
                    device.type == 'cuda'
                    and ckpt.get(
                        'cuda_rngs'
                    )
                ):

                    for i, rng in enumerate(
                        ckpt[
                            'cuda_rngs'
                        ]
                    ):

                        try:

                            torch.cuda.set_rng_state(
                                rng,
                                i,
                            )

                        except Exception:
                            pass


            except Exception:

                logger.warning(
                    "  RNG state could not be restored."
                )


            training_history = ckpt.get(
                'training_history',
                [],
            )


            start_epoch = ckpt.get(
                'epoch',
                0,
            ) + 1


            logger.info(
                f"  Resuming from epoch "
                f"{start_epoch}"
            )


        except Exception as e:

            logger.warning(
                f"  Unable to load checkpoint: {e}. "
                f"Starting from scratch."
            )


    logger.info(
        "\n[Sanity check] "
        "Evaluating warm start (epoch 0)..."
    )


    reset_eval_debug()


    val0 = evaluate_task_b_fashion(
        model,
        valid_data,
        r2u,
    )


    test0 = evaluate_task_b_fashion(
        model,
        test_data,
        r2u,
    )


    r20_val0   = val0[
        'Recall@20'
    ]

    ndcg_val0  = val0[
        'NDCG@20'
    ]

    r20_test0  = test0[
        'Recall@20'
    ]

    ndcg_test0 = test0[
        'NDCG@20'
    ]


    logger.info(
        f"  Epoch 0 | "
        f"Valid: R@20={r20_val0:.4f} "
        f"NDCG@20={ndcg_val0:.4f} | "
        f"Test: R@20={r20_test0:.4f} "
        f"NDCG@20={ndcg_test0:.4f}"
    )


    logger.info(
        f"  (Standalone RecBole baseline: "
        f"R@20={BASELINE_RECALL20}, "
        f"NDCG@20={BASELINE_NDCG20})"
    )


    # ===================================================================== #
    # STEP 9: Training loop
    # ===================================================================== #

    logger.info(
        f"\n[9/10] Alternate training — "
        f"{config['training']['epochs']} epochs..."
    )


    best_r20_val    = -1.0
    best_ndcg_val   = -1.0
    best_epoch_val  = 0
    best_r20_test   = -1.0
    best_epoch_test = 0


    num_epochs = config[
        'training'
    ][
        'epochs'
    ]


    eval_step = config[
        'training'
    ].get(
        'eval_step',
        1,
    )


    for epoch in range(
        start_epoch,
        num_epochs + 1,
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
                ],
        }


        training_history.append(
            entry
        )


        logger.info(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"LossA={stats['loss_a_mean']:.4f} | "
            f"LossB={stats['loss_b_mean']:.4f} | "
            f"batchA={stats['batches_a']} "
            f"batchB={stats['batches_b']} | "
            f"time={stats['time_sec']:.1f}s"
        )


        if epoch % eval_step == 0:


            reset_eval_debug()


            val_m = evaluate_task_b_fashion(
                model,
                valid_data,
                r2u,
            )


            r20_v = val_m[
                'Recall@20'
            ]

            ndcg_v = val_m[
                'NDCG@20'
            ]


            entry[
                'val_recall_20'
            ] = r20_v


            entry[
                'val_ndcg_20'
            ] = ndcg_v


            reset_eval_debug()


            test_m = evaluate_task_b_fashion(
                model,
                test_data,
                r2u,
            )


            r20_t = test_m[
                'Recall@20'
            ]

            ndcg_t = test_m[
                'NDCG@20'
            ]


            entry[
                'test_recall_20'
            ] = r20_t


            entry[
                'test_ndcg_20'
            ] = ndcg_t


            logger.info(
                f"  ==> Valid: "
                f"R@20={r20_v:.4f} "
                f"NDCG@20={ndcg_v:.4f} | "
                f"Test: "
                f"R@20={r20_t:.4f} "
                f"NDCG@20={ndcg_t:.4f}"
            )


            if r20_v > best_r20_val:


                best_r20_val   = r20_v
                best_ndcg_val  = ndcg_v
                best_epoch_val = epoch


                torch.save(
                    model.state_dict(),
                    os.path.join(
                        out_dir,
                        "best_model.pt",
                    ),
                )


                logger.info(
                    f"  => NEW BEST VALID | "
                    f"epoch={epoch} | "
                    f"R@20={r20_v:.4f} "
                    f"NDCG@20={ndcg_v:.4f} | "
                    f"corresponding test: "
                    f"R@20={r20_t:.4f} "
                    f"NDCG@20={ndcg_t:.4f}"
                )


            if r20_t > best_r20_test:


                best_r20_test   = r20_t
                best_epoch_test = epoch


                torch.save(
                    model.state_dict(),
                    os.path.join(
                        out_dir,
                        "best_model_on_test.pt",
                    ),
                )


                logger.info(
                    f"  => NEW BEST TEST | "
                    f"epoch={epoch} | "
                    f"R@20={r20_t:.4f} "
                    f"NDCG@20={ndcg_t:.4f}"
                )


        with open(
            os.path.join(
                out_dir,
                "training_history.json",
            ),
            "w",
        ) as f:


            json.dump(
                training_history,
                f,
                indent=4,
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
                torch.get_rng_state(),
        }


        if torch.cuda.is_available():


            try:

                ckpt[
                    'cuda_rngs'
                ] = [
                    torch.cuda.get_rng_state(i)
                    for i in range(
                        torch.cuda.device_count()
                    )
                ]


            except Exception:

                ckpt[
                    'cuda_rngs'
                ] = None


        torch.save(
            ckpt,
            os.path.join(
                out_dir,
                'checkpoint_latest.pth',
            ),
        )


    torch.save(
        model.state_dict(),
        os.path.join(
            out_dir,
            "last_model.pt",
        ),
    )


    # ===================================================================== #
    # STEP 10: Final evaluation
    # ===================================================================== #

    logger.info(
        "\n[10/10] Final evaluation on the test set "
        "using best_model.pt..."
    )


    best_model_path = os.path.join(
        out_dir,
        "best_model.pt",
    )


    if os.path.exists(
        best_model_path
    ):


        model.load_state_dict(
            torch.load(
                best_model_path,
                map_location=device,
                weights_only=False,
            )
        )


        logger.info(
            f"  Loaded best_model.pt "
            f"(best validation epoch "
            f"{best_epoch_val})"
        )


    else:

        logger.warning(
            "  best_model.pt not found — "
            "using last_model.pt"
        )


    reset_eval_debug()


    final_test = evaluate_task_b_fashion(
        model,
        test_data,
        r2u,
    )


    r20_final = final_test[
        'Recall@20'
    ]

    ndcg_final = final_test[
        'NDCG@20'
    ]


    logger.info("=" * 70)

    logger.info(
        "FINAL RESULTS — TEST SET — "
        "COMPLETE 5-RELATION KG"
    )

    logger.info("=" * 70)


    logger.info(
        f"  Recall@20 : {r20_final:.4f} "
        f"(standalone SASRec baseline: "
        f"{BASELINE_RECALL20})"
    )


    logger.info(
        f"  NDCG@20   : {ndcg_final:.4f} "
        f"(standalone SASRec baseline: "
        f"{BASELINE_NDCG20})"
    )


    final_results = {

        "variant":
            "full5rel",

        "kg_train_used":
            KG_TRAIN,

        "transe_checkpoint_used":
            TRANSE_PATH,

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

        "baseline_sasrec_recall_20":
            BASELINE_RECALL20,

        "baseline_sasrec_ndcg_20":
            BASELINE_NDCG20,

        "warmstart_val_recall_20":
            r20_val0,

        "warmstart_val_ndcg_20":
            ndcg_val0,

        "warmstart_test_recall_20":
            r20_test0,

        "warmstart_test_ndcg_20":
            ndcg_test0,

        "hyperparameters": {

            "learning_rate":
                config[
                    'training'
                ][
                    'learning_rate'
                ],

            "batch_size_kge":
                config[
                    'training'
                ][
                    'batch_size_kge'
                ],

            "batch_size_sasrec":
                config[
                    'training'
                ][
                    'batch_size_sasrec'
                ],

            "margin_loss":
                config[
                    'training'
                ][
                    'margin_loss'
                ],

            "hidden_dropout":
                config[
                    'model'
                ][
                    'hidden_dropout'
                ],

            "attn_dropout":
                config[
                    'model'
                ][
                    'attn_dropout'
                ],

            "kge_dim":
                config[
                    'model'
                ][
                    'kge_dim'
                ],

            "n_layers":
                config[
                    'model'
                ][
                    'n_layers'
                ],

            "n_heads":
                config[
                    'model'
                ][
                    'n_heads'
                ],
        },
    }


    with open(
        os.path.join(
            out_dir,
            "best_metrics.json",
        ),
        "w",
    ) as f:


        json.dump(
            final_results,
            f,
            indent=4,
        )


    logger.info(
        f"\nTraining completed. "
        f"Results saved in {out_dir}"
    )


if __name__ == '__main__':
    main()
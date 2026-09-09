import os
import sys
import ast
import logging
import warnings

import torch
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

_original_load = torch.load

def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)

torch.load = _patched_load


from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import SASRec
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
BASE_DIR = os.path.abspath(os.path.join(PARENT_DIR, ".."))

DATA_PATH = os.path.join(BASE_DIR, "data", "recbole")

RESULTS_CSV_CANDIDATES = [
    os.path.join(BASE_DIR, "results", "sasrec_hpo", "hpo_results_sasrec_fashion.csv"),
    os.path.join(BASE_DIR, "results", "hpo_results_sasrec_fashion.csv"),
]

TRIAL_ID = 17
CHECKPOINT_CANDIDATES = [
    os.path.join(BASE_DIR, "results", "sasrec_hpo", f"trial_{TRIAL_ID:03d}", "best_valid.pth"),
    os.path.join(BASE_DIR, "results", f"trial_{TRIAL_ID:03d}", "best_valid.pth"),
]

EXPECTED_VALID_NDCG = 0.1086
EXPECTED_TEST_NDCG = 0.0889
EXPECTED_TEST_RECALL = 0.0964

TOL = 5e-4


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def first_existing_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def parse_float(x):
    try:
        return float(x)
    except Exception:
        return x


def load_trial_row():
    csv_path = first_existing_path(RESULTS_CSV_CANDIDATES)

    if csv_path is None:
        raise FileNotFoundError(
            "I can't find hpo_results_sasrec_fashion.csv.\n"
            "Without this CSV I can't know for sure the hyperparameters"
            "of trial 17, especially n_heads.\n\n"
            "Searched paths:\n"
            + "\n".join(f" - {p}" for p in RESULTS_CSV_CANDIDATES)
            + "\n\n"
            "Solution: Copy the HPO CSV into results/sasrec_hpo or "
            "paste the full row from trial 17 into me."
        )

    logger.info(f"CSV HPO : {csv_path}")

    df = pd.read_csv(csv_path)

    if "trial" not in df.columns:
        raise ValueError(
            f"The CSV {csv_path} does not contain the column 'trial'. "
            f"Columns found: {list(df.columns)}"
        )

    row = df[df["trial"].astype(int) == TRIAL_ID]

    if len(row) == 0:
        raise ValueError(
            f"Trial {TRIAL_ID} not found in CSV {csv_path}."
        )

    if len(row) > 1:
        logger.warning(f"Found more than one row for trial {TRIAL_ID}; using the first one.")

    return row.iloc[0].to_dict(), csv_path


def build_params_from_row(row):
    """
    Builds the parameters for the SASRec model from a row in the HPO CSV.
    """

    required = [
        "n_layers",
        "n_heads",
        "learning_rate",
        "hidden_dropout_prob",
        "attn_dropout_prob",
        "weight_decay",
        "loss_type",
        "neg_candidate_num",
    ]

    missing = [c for c in required if c not in row]
    if missing:
        raise ValueError(
            f"The CSV is missing required columns to rebuild the trial: {missing}\n"
            f"Available columns: {list(row.keys())}"
        )

    params = {
        "n_layers": int(row["n_layers"]),
        "n_heads": int(row["n_heads"]),
        "learning_rate": float(row["learning_rate"]),
        "hidden_dropout_prob": float(row["hidden_dropout_prob"]),
        "attn_dropout_prob": float(row["attn_dropout_prob"]),
        "weight_decay": float(row["weight_decay"]),
        "loss_type": str(row["loss_type"]),
        "neg_candidate_num": row["neg_candidate_num"],
    }

    if str(params["neg_candidate_num"]) not in {"N/A", "nan", "None"}:
        try:
            params["neg_candidate_num"] = int(float(params["neg_candidate_num"]))
        except Exception:
            pass

    return params


def build_config(params):
    """
    Builds the config for the SASRec model.
    """

    config_dict = {
        "model": "SASRec",

        "dataset": "fashion",
        "data_path": DATA_PATH,
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "TIME_FIELD": "timestamp",
        "load_col": {
            "inter": ["user_id", "item_id", "timestamp"],
        },
        "field_separator": "\t",
        "eval_args": {
            "split": {"LS": "valid_and_test"},
            "order": "TO",
            "group_by": "user",
            "mode": {
                "valid": "full",
                "test": "full",
            },
        },


        "hidden_size": 64,
        "MAX_ITEM_LIST_LENGTH": 50,
        "n_layers": params["n_layers"],
        "n_heads": params["n_heads"],
        "hidden_dropout_prob": params["hidden_dropout_prob"],
        "attn_dropout_prob": params["attn_dropout_prob"],
        "loss_type": params["loss_type"],
        "learning_rate": params["learning_rate"],
        "weight_decay": params["weight_decay"],
        "epochs": 200,
        "train_batch_size": 256,
        "eval_batch_size": 512,
        "stopping_step": 10,
        "metrics": ["Recall", "NDCG"],
        "topk": [20],
        "valid_metric": "NDCG@20",
        "reproducibility": True,
        "seed": 2020,
        "use_gpu": torch.cuda.is_available(),
        "show_progress": False,
    }

    if params["loss_type"] == "CE":
        config_dict["train_neg_sample_args"] = None
    else:
        cand = int(params["neg_candidate_num"])
        config_dict["train_neg_sample_args"] = {
            "distribution": "uniform",
            "sample_num": cand,
            "alpha": 1.0,
            "dynamic": False,
            "candidate_num": 0,
        }

    return Config(model="SASRec", config_dict=config_dict)


def load_checkpoint_state_dict(path, device):

    ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]

    return ckpt


def inspect_checkpoint(state_dict):

    logger.info("\nCheckpoint tensors:")

    key_shapes = [
        "item_embedding.weight",
        "position_embedding.weight",
        "LayerNorm.weight",
        "LayerNorm.bias",
    ]

    for k in key_shapes:
        if k in state_dict:
            logger.info(f"  {k:35s}: {tuple(state_dict[k].shape)}")
        else:
            logger.warning(f"  {k:35s}: MISSING")

    n_layers = sorted({
        int(k.split(".")[2])
        for k in state_dict.keys()
        if k.startswith("trm_encoder.layer.")
    })

    logger.info(f"  transformer layers found       : {n_layers}")
    logger.info(f"  number of transformer layers   : {len(n_layers)}")

    if "item_embedding.weight" in state_dict:
        w = state_dict["item_embedding.weight"]
        logger.info(f"  item_embedding real items      : {w.shape[0] - 1:,}")
        logger.info(f"  hidden size from checkpoint    : {w.shape[1]}")


def evaluate_with_valid_epoch(config, model, valid_data, test_data):


    trainer = Trainer(config, model)

    _, valid_result = trainer._valid_epoch(
        valid_data,
        show_progress=False,
    )

    _, test_result = trainer._valid_epoch(
        test_data,
        show_progress=False,
    )

    return valid_result, test_result


def compare_metric(name, reproduced, expected):
    diff = reproduced - expected
    ok = abs(diff) <= TOL

    status = "PASS" if ok else "FAIL"

    logger.info(
        f"  {name:25s} expected={expected:.4f} "
        f"reproduced={reproduced:.4f} diff={diff:+.6f} [{status}]"
    )

    return ok


def main():
    logger.info("=" * 70)
    logger.info("VERIFY SASREC METRICS — FASHION")
    logger.info("=" * 70)

    # ------------------------------------------------------------------ #
    # 1. Load trial params
    # ------------------------------------------------------------------ #
    logger.info(f"\n[1/6] Reading trial parameters {TRIAL_ID}...")

    row, csv_path = load_trial_row()
    params = build_params_from_row(row)

    logger.info("Trial parameters:")
    for k, v in params.items():
        logger.info(f"  {k:25s}: {v}")

    logger.info("\nExpected metrics from CSV / log:")
    for col in [
        "best_valid_ndcg",
        "best_valid_recall",
        "test_from_best_valid_ndcg",
        "test_from_best_valid_recall",
    ]:
        if col in row:
            logger.info(f"  {col:30s}: {row[col]}")


    logger.info(f"\n[2/6] Locating checkpoint best_valid.pth...")

    ckpt_path = first_existing_path(CHECKPOINT_CANDIDATES)

    if ckpt_path is None:
        raise FileNotFoundError(
            "Can't find best_valid.pth for test 017.\n"
            "Routes searched:\n"
            + "\n".join(f"  - {p}" for p in CHECKPOINT_CANDIDATES)
        )

    logger.info(f"Checkpoint found: {ckpt_path}")

    logger.info("\n[3/6] Reconstruction config/dataset/dataloader RecBole...")

    config = build_config(params)

    init_seed(
        config["seed"],
        config["reproducibility"],
    )
    init_logger(config)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    logger.info(f"Dataset item_num including PAD : {dataset.item_num:,}")
    logger.info(f"Dataset real items             : {dataset.item_num - 1:,}")
    logger.info(f"Dataset user_num               : {dataset.user_num:,}")
    logger.info(f"Dataset inter_num              : {dataset.inter_num:,}")

    logger.info("\nEvaluation protocol:")
    logger.info(f"  split    : {config['eval_args']['split']}")
    logger.info(f"  order    : {config['eval_args']['order']}")
    logger.info(f"  group_by : {config['eval_args']['group_by']}")
    logger.info(f"  mode     : {config['eval_args']['mode']}")


    logger.info("\n[4/6] SASRec construction and weight loading...")

    device = config["device"]
    logger.info(f"Device: {device}")

    model = SASRec(config, train_data.dataset).to(device)

    state_dict = load_checkpoint_state_dict(
        ckpt_path,
        device=device,
    )

    inspect_checkpoint(state_dict)

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if missing:
        logger.error(f"Missing keys: {missing}")
    if unexpected:
        logger.error(f"Unexpected keys: {unexpected}")

    assert len(missing) == 0, f"Missing keys nel checkpoint: {missing}"
    assert len(unexpected) == 0, f"Unexpected keys nel checkpoint: {unexpected}"

    model.eval()

    logger.info("PASS: Checkpoint loaded successfully in SASRec template.")


    logger.info("\n[5/6] Evaluation with Trainer._valid_epoch...")

    valid_result, test_result = evaluate_with_valid_epoch(
        config=config,
        model=model,
        valid_data=valid_data,
        test_data=test_data,
    )

    valid_ndcg = float(valid_result.get("ndcg@20", 0.0))
    valid_recall = float(valid_result.get("recall@20", 0.0))

    test_ndcg = float(test_result.get("ndcg@20", 0.0))
    test_recall = float(test_result.get("recall@20", 0.0))

    logger.info("\nResults reproduced:")
    logger.info(f"  valid_result: {valid_result}")
    logger.info(f"  test_result : {test_result}")

    # ------------------------------------------------------------------ #
    # 6. Compare
    # ------------------------------------------------------------------ #
    logger.info("\n[6/6] Comparison with expected values...")

    ok_valid_ndcg = compare_metric(
        "valid ndcg@20",
        valid_ndcg,
        EXPECTED_VALID_NDCG,
    )

    ok_test_ndcg = compare_metric(
        "test ndcg@20",
        test_ndcg,
        EXPECTED_TEST_NDCG,
    )

    ok_test_recall = compare_metric(
        "test recall@20",
        test_recall,
        EXPECTED_TEST_RECALL,
    )

    logger.info("\n" + "=" * 70)

    if ok_valid_ndcg and ok_test_ndcg and ok_test_recall:
        logger.info("VERDICT: PASS — SASRec best_valid.pth riproduce le metriche HPO.")
        logger.info("I pesi SASRec sono validi per il warm start / confronto Task B.")
    else:
        logger.error("VERDICT: FAIL - SASRec metrics do not match HPO.")
        logger.error("Possible causes:")
        logger.error(" 1. Trial parameters 17 incorrect.")
        logger.error(" 2. CSV HPO does not match the copied checkpoint.")
        logger.error(" 3. Checkpoint best_valid.pth different from the one used in the log.")
        logger.error(" 4. RecBole/dataset fashion.inter version not identical.")
        sys.exit(1)

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
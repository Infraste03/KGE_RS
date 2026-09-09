"""
run_transe_ablation_batch.py

Trains TransE on all variants generated from ablation_manifest.json,
saving separate outputs for each run.

IMPORTANT: the vocabulary (entity_to_id, relation_to_id) is fixed and
always computed from the original, unfiltered KG. This ensures that
num_entities and num_relations remain identical across all variants,
which is required for the Step 4 warm start.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory


# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

MANIFEST_DIR = BASE_DIR / "data" / "processed" / "ablation"
MANIFEST_PATH = MANIFEST_DIR / "ablation_manifest.json"

VALID_PATH = BASE_DIR / "data" / "processed" / "taskA_valid.tsv"
TEST_PATH = BASE_DIR / "data" / "processed" / "taskA_test.tsv"
ORIGINAL_KG_TRAIN = BASE_DIR / "data" / "processed" / "kg_train.tsv"

OUTPUT_ROOT = BASE_DIR / "results" / "ablation_transe"

# Add the directory containing load_kg.py to the Python path.
sys.path.insert(0, str(BASE_DIR / "B2B_alternate_learning"))

from load_kg import load_kg  # noqa: E402


RANDOM_SEED = 42
TRAINING_EPOCHS_FINAL = 300
BATCH_SIZE = 1024

RELATION_LABEL = "compatible_with"
HITS_AT_K = [1, 3, 10]


BEST_PARAMS = {
    "embedding_dim": 400,
    "learning_rate": 0.0008798929749689024,
    "num_negs_per_pos": 128,
    "loss": "marginranking",
    "loss_margin": 5.191058165461712,
}


EARLY_STOPPING_KWARGS = {
    "frequency": 5,
    "patience": 3,
    "metric": "inverse_harmonic_mean_rank",
    "relative_delta": 0.002,
}


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def setup_root_logger() -> logging.Logger:
    logger = logging.getLogger("ablation_transe")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(message)s")
        )
        logger.addHandler(stream_handler)

    return logger


def setup_variant_logger(
    variant_name: str,
    variant_dir: Path,
) -> logging.Logger:
    logger = logging.getLogger(f"ablation_transe.{variant_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = True

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    log_path = variant_dir / f"run_{variant_name}.log"

    file_handler = logging.FileHandler(
        log_path,
        mode="w",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(message)s")
    )

    logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_manifest(manifest_path: Path) -> dict:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_ablation_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    candidate = (MANIFEST_DIR / path.name).resolve()
    if candidate.exists():
        return candidate

    candidate = (BASE_DIR / path).resolve()
    if candidate.exists():
        return candidate

    return candidate


def build_fixed_vocab(logger: logging.Logger):
    """
    Build a fixed vocabulary (entity_to_id, relation_to_id) using the
    original, unfiltered KG files.

    The same mappings are reused for every ablation variant so that
    num_entities=24,957 and num_relations=5 remain unchanged.

    The alphabetical ordering is therefore consistent with the mapping used
    by unified_id_space.py during the Step 4 warm start.
    """

    kg = load_kg(
        train_path=str(ORIGINAL_KG_TRAIN),
        valid_path=str(VALID_PATH),
        test_path=str(TEST_PATH),
        verbose=False,
    )

    logger.info(
        f"Fixed vocabulary from original graph: "
        f"{kg['num_entities']:,} entities, "
        f"{kg['num_relations']} relations"
    )

    return kg["entity_to_id"], kg["relation_to_id"]


def load_triples_factories(
    kg_train_path: Path,
    valid_path: Path,
    test_path: Path,
    entity_to_id: dict,
    relation_to_id: dict,
    logger: logging.Logger,
):
    train_df = pd.read_csv(kg_train_path, sep="\t")
    valid_df = pd.read_csv(valid_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t")

    train_arr = train_df[["head", "relation", "tail"]].values
    valid_arr = valid_df[["head", "relation", "tail"]].values
    test_arr = test_df[["head", "relation", "tail"]].values

    # Explicitly provide entity_to_id and relation_to_id instead of
    # auto-inferring them from each filtered KG.
    # This guarantees fixed num_entities and num_relations across variants.
    train_factory = TriplesFactory.from_labeled_triples(
        train_arr,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )

    valid_factory = TriplesFactory.from_labeled_triples(
        valid_arr,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )

    test_factory = TriplesFactory.from_labeled_triples(
        test_arr,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )

    logger.info(f"  Entities : {train_factory.num_entities:,}")
    logger.info(f"  Relations: {train_factory.num_relations:,}")

    logger.info(
        f"  Triples  : train={train_factory.num_triples:,} (filtered), "
        f"valid={valid_factory.num_triples:,}, "
        f"test={test_factory.num_triples:,}"
    )

    return train_factory, valid_factory, test_factory


def build_type_index(train_factory):
    type_to_ids = {
        "item": [],
        "machine": [],
        "client": [],
        "model": [],
    }

    for label, eid in train_factory.entity_to_id.items():
        prefix = label.split("_")[0]

        if prefix in type_to_ids:
            type_to_ids[prefix].append(eid)

    return {
        k: torch.LongTensor(v)
        for k, v in type_to_ids.items()
    }


def evaluate_custom(
    model,
    test_factory,
    train_factory,
    valid_factory,
    type_index,
    mode,
    relation_label=RELATION_LABEL,
):
    device = next(model.parameters()).device
    relation_id = train_factory.relation_to_id[relation_label]
    num_entities = train_factory.num_entities

    if mode == "type_constrained":
        tail_mask = torch.full(
            (num_entities,),
            float("-inf"),
            device=device,
        )
        tail_mask[type_index["machine"].to(device)] = 0.0

        head_mask = torch.full(
            (num_entities,),
            float("-inf"),
            device=device,
        )
        head_mask[type_index["item"].to(device)] = 0.0

    else:
        tail_mask = torch.zeros(
            num_entities,
            device=device,
        )
        head_mask = torch.zeros(
            num_entities,
            device=device,
        )

    head_to_known_tails = {}
    tail_to_known_heads = {}

    for fac in [
        train_factory,
        valid_factory,
        test_factory,
    ]:
        for h, r, t in fac.mapped_triples.tolist():
            if r != relation_id:
                continue

            head_to_known_tails.setdefault(h, set()).add(t)
            tail_to_known_heads.setdefault(t, set()).add(h)

    test_triples = test_factory.mapped_triples
    test_triples = test_triples[
        test_triples[:, 1] == relation_id
    ]

    model.eval()

    tail_ranks = []
    head_ranks = []

    with torch.no_grad():
        for h, r, t in test_triples.tolist():

            scores = model.score_t(
                torch.tensor(
                    [[h, r]],
                    device=device,
                )
            ).squeeze(0)

            scores = scores + tail_mask

            for known_tail in head_to_known_tails.get(h, set()):
                if known_tail != t:
                    scores[known_tail] = float("-inf")

            tail_ranks.append(
                (scores > scores[t]).sum().item() + 1
            )

            scores = model.score_h(
                torch.tensor(
                    [[r, t]],
                    device=device,
                )
            ).squeeze(0)

            scores = scores + head_mask

            for known_head in tail_to_known_heads.get(t, set()):
                if known_head != h:
                    scores[known_head] = float("-inf")

            head_ranks.append(
                (scores > scores[h]).sum().item() + 1
            )

    def metrics(ranks):
        ranks = torch.tensor(ranks).float()

        out = {
            "MRR": (1.0 / ranks).mean().item(),
            "MR": ranks.mean().item(),
        }

        for k in HITS_AT_K:
            out[f"Hits@{k}"] = (
                (ranks <= k).float().mean().item()
            )

        return out

    tail_metrics = metrics(tail_ranks)
    head_metrics = metrics(head_ranks)

    avg_metrics = {
        key: (
            tail_metrics[key] + head_metrics[key]
        ) / 2
        for key in tail_metrics
    }

    return {
        "tail": tail_metrics,
        "head": head_metrics,
        "avg": avg_metrics,
    }


def flatten_metrics(
    prefix: str,
    metrics: dict,
) -> dict:
    avg = metrics["avg"]

    return {
        f"{prefix}_mrr": avg["MRR"],
        f"{prefix}_mr": avg["MR"],
        f"{prefix}_hits@1": avg["Hits@1"],
        f"{prefix}_hits@3": avg["Hits@3"],
        f"{prefix}_hits@10": avg["Hits@10"],
    }


def build_run_context(
    variant_name: str,
    spec: dict,
    best_params: dict,
) -> dict:
    kept = [
        str(x)
        for x in spec.get("relations_kept", [])
    ]

    excluded = [
        str(x)
        for x in spec.get("relations_excluded", [])
    ]

    return {
        "variant": variant_name,
        "feature_count": len(kept),
        "features_kept": kept,
        "features_excluded": excluded,
        "feature_signature": (
            "|".join(kept)
            if kept
            else "none"
        ),
        "hyperparams": best_params,
    }


def log_run_context(
    logger: logging.Logger,
    context: dict,
) -> None:
    logger.info("=== RUN CONTEXT ===")

    logger.info(
        "variant           : %s",
        context["variant"],
    )

    logger.info(
        "feature_count     : %d",
        context["feature_count"],
    )

    logger.info(
        "features_kept     : %s",
        (
            ", ".join(context["features_kept"])
            if context["features_kept"]
            else "none"
        ),
    )

    logger.info(
        "features_excluded : %s",
        (
            ", ".join(context["features_excluded"])
            if context["features_excluded"]
            else "none"
        ),
    )

    logger.info(
        "feature_signature : %s",
        context["feature_signature"],
    )

    for key, value in context["hyperparams"].items():
        logger.info(
            "hyperparam %-15s: %s",
            key,
            value,
        )


# ---------------------------------------------------------------------
# Variant run
# ---------------------------------------------------------------------

def build_pipeline_kwargs(
    train_factory,
    valid_factory,
    device,
    best_params,
):
    return {
        "training": train_factory,
        "validation": valid_factory,
        "testing": valid_factory,
        "model": "TransE",
        "model_kwargs": {
            "embedding_dim": best_params["embedding_dim"]
        },
        "loss": best_params["loss"],
        "loss_kwargs": {
            "margin": best_params["loss_margin"]
        },
        "optimizer": "Adam",
        "optimizer_kwargs": {
            "lr": best_params["learning_rate"]
        },
        "training_kwargs": {
            "num_epochs": TRAINING_EPOCHS_FINAL,
            "batch_size": BATCH_SIZE,
        },
        "negative_sampler": "basic",
        "negative_sampler_kwargs": {
            "num_negs_per_pos": best_params[
                "num_negs_per_pos"
            ]
        },
        "stopper": "early",
        "stopper_kwargs": EARLY_STOPPING_KWARGS,
        "random_seed": RANDOM_SEED,
        "device": device,
        "use_tqdm": True,
    }


def run_one_variant(
    variant_name: str,
    spec: dict,
    root_logger: logging.Logger,
):
    variant_dir = OUTPUT_ROOT / variant_name
    variant_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    variant_logger = setup_variant_logger(
        variant_name,
        variant_dir,
    )

    kg_train_path = resolve_ablation_path(
        spec["kg_train_ablation"]
    )

    if not kg_train_path.exists():
        raise FileNotFoundError(
            f"File not found: {kg_train_path}"
        )

    variant_logger.info("=" * 90)
    variant_logger.info(
        f"VARIANT: {variant_name}"
    )
    variant_logger.info(
        f"KG TRAIN: {kg_train_path}"
    )
    variant_logger.info(
        f"KIND    : {spec.get('kind', 'unknown')}"
    )
    variant_logger.info("=" * 90)

    context = build_run_context(
        variant_name,
        spec,
        BEST_PARAMS,
    )

    log_run_context(
        variant_logger,
        context,
    )

    context_path = (
        variant_dir
        / f"context_{variant_name}.json"
    )

    with open(
        context_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            context,
            f,
            indent=2,
        )

    # Build the fixed vocabulary from the original KG.
    entity_to_id, relation_to_id = build_fixed_vocab(
        variant_logger
    )

    (
        train_factory,
        valid_factory,
        test_factory,
    ) = load_triples_factories(
        kg_train_path=kg_train_path,
        valid_path=VALID_PATH,
        test_path=TEST_PATH,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        logger=variant_logger,
    )

    # Sanity check: fail immediately if the fixed vocabulary
    # is not preserved.
    assert (
        train_factory.num_entities
        == len(entity_to_id)
    ), "num_entities mismatch!"

    assert (
        train_factory.num_relations
        == len(relation_to_id)
    ), "num_relations mismatch!"

    variant_logger.info(
        f"Sanity check OK: "
        f"num_entities={train_factory.num_entities}, "
        f"num_relations={train_factory.num_relations} "
        f"(fixed across all variants)"
    )

    type_index = build_type_index(
        train_factory
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    variant_logger.info(
        f"Device: {device}"
    )

    if device == "cuda":
        variant_logger.info(
            f"GPU   : "
            f"{torch.cuda.get_device_name(0)}"
        )

    variant_logger.info(
        "Best parameters used:"
    )

    for key, value in BEST_PARAMS.items():
        variant_logger.info(
            f"  {key:18s} = {value}"
        )

    t0 = time.time()

    result = pipeline(
        **build_pipeline_kwargs(
            train_factory=train_factory,
            valid_factory=valid_factory,
            device=device,
            best_params=BEST_PARAMS,
        )
    )

    elapsed = time.time() - t0

    variant_logger.info(
        f"Training completed in {elapsed:.0f}s"
    )

    naive_results = evaluate_custom(
        result.model,
        test_factory,
        train_factory,
        valid_factory,
        type_index,
        mode="naive",
    )

    typed_results = evaluate_custom(
        result.model,
        test_factory,
        train_factory,
        valid_factory,
        type_index,
        mode="type_constrained",
    )

    best_model_path = (
        variant_dir
        / f"best_model_{variant_name}.pt"
    )

    best_metrics_path = (
        variant_dir
        / f"best_metrics_{variant_name}.json"
    )

    config_path = (
        variant_dir
        / f"run_config_{variant_name}.json"
    )

    torch.save(
        result.model.state_dict(),
        best_model_path,
    )

    with open(
        best_metrics_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "variant": variant_name,
                "kind": spec.get("kind"),
                "kg_train_ablation": str(
                    kg_train_path
                ),
                "naive": naive_results,
                "type_constrained": typed_results,
                "elapsed_seconds": elapsed,
            },
            f,
            indent=2,
        )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "variant": variant_name,
                "kind": spec.get("kind"),
                "kg_train_ablation": str(
                    kg_train_path
                ),
                "relations_kept": spec.get(
                    "relations_kept",
                    [],
                ),
                "relations_excluded": spec.get(
                    "relations_excluded",
                    [],
                ),
                "feature_count": len(
                    spec.get(
                        "relations_kept",
                        [],
                    )
                ),
                "feature_signature": (
                    "|".join(
                        str(x)
                        for x in spec.get(
                            "relations_kept",
                            [],
                        )
                    )
                    if spec.get(
                        "relations_kept"
                    )
                    else "none"
                ),
                "best_params": BEST_PARAMS,
                "random_seed": RANDOM_SEED,
                "training_epochs_final":
                    TRAINING_EPOCHS_FINAL,
                "batch_size": BATCH_SIZE,
            },
            f,
            indent=2,
        )

    variant_logger.info(
        "FINAL COMPARISON TABLE "
        "(TEST set, averaged head+tail)"
    )

    variant_logger.info(
        f"{'Model':10s} "
        f"{'Mode':18s} "
        f"{'MRR':>8s} "
        f"{'H@1':>8s} "
        f"{'H@3':>8s} "
        f"{'H@10':>8s}"
    )

    variant_logger.info(
        "-" * 60
    )

    for mode, res in [
        ("naive", naive_results),
        ("type_constrained", typed_results),
    ]:
        avg = res["avg"]

        variant_logger.info(
            f"{'TransE':10s} "
            f"{mode:18s} "
            f"{avg['MRR']:>8.4f} "
            f"{avg['Hits@1']:>8.4f} "
            f"{avg['Hits@3']:>8.4f} "
            f"{avg['Hits@10']:>8.4f}"
        )

    variant_logger.info(
        f"Best model saved to: "
        f"{best_model_path}"
    )

    variant_logger.info(
        f"Metrics saved to   : "
        f"{best_metrics_path}"
    )

    variant_logger.info(
        f"Config saved to    : "
        f"{config_path}"
    )

    row = {
        "variant": variant_name,
        "kind": spec.get("kind", ""),
        "kg_train_ablation": str(
            kg_train_path
        ),
        "status": "ok",
        "elapsed_seconds": round(
            elapsed,
            4,
        ),
        "best_model_path": str(
            best_model_path
        ),
        "best_metrics_path": str(
            best_metrics_path
        ),
        "log_path": str(
            variant_dir
            / f"run_{variant_name}.log"
        ),
    }

    row.update(
        flatten_metrics(
            "naive",
            naive_results,
        )
    )

    row.update(
        flatten_metrics(
            "type",
            typed_results,
        )
    )

    # Release memory before processing the next variant.
    del (
        result,
        train_factory,
        valid_factory,
        test_factory,
    )

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help=(
            "Name of a single variant to run "
            "(e.g. loo_no_bought). "
            "If omitted, all variants are executed."
        ),
    )

    args = parser.parse_args()

    root_logger = setup_root_logger()

    root_logger.info(
        "Loading manifest..."
    )

    manifest = load_manifest(
        MANIFEST_PATH
    )

    if args.variant is not None:

        if args.variant not in manifest:
            raise ValueError(
                f"Variant '{args.variant}' "
                f"not found in the manifest. "
                f"Available variants: "
                f"{list(manifest.keys())}"
            )

        manifest = {
            args.variant:
                manifest[args.variant]
        }

        root_logger.info(
            f"Single-variant mode: "
            f"{args.variant}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    root_logger.info(
        f"Manifest path: {MANIFEST_PATH}"
    )

    root_logger.info(
        f"Output root  : {OUTPUT_ROOT}"
    )

    root_logger.info(
        f"Variants     : {len(manifest)}"
    )

    rows = []
    failures = {}

    for variant_name, spec in manifest.items():

        # Skip variants that have already been completed.
        # This is useful when resuming an interrupted run.
        variant_dir = (
            OUTPUT_ROOT
            / variant_name
        )

        if (
            variant_dir
            / f"best_metrics_{variant_name}.json"
        ).exists():
            root_logger.info(
                f"Skip (already completed): "
                f"{variant_name}"
            )
            continue

        root_logger.info(
            f"Starting variant: "
            f"{variant_name}"
        )

        try:
            row = run_one_variant(
                variant_name=variant_name,
                spec=spec,
                root_logger=root_logger,
            )

            rows.append(row)

        except Exception as exc:
            failures[
                variant_name
            ] = str(exc)

            root_logger.exception(
                f"Variant failed: "
                f"{variant_name} -> {exc}"
            )

            rows.append(
                {
                    "variant": variant_name,
                    "kind": spec.get(
                        "kind",
                        "",
                    ),
                    "kg_train_ablation": str(
                        spec.get(
                            "kg_train_ablation",
                            "",
                        )
                    ),
                    "status": "failed",
                    "error": str(exc),
                    "elapsed_seconds": None,
                    "best_model_path": "",
                    "best_metrics_path": "",
                    "log_path": str(
                        OUTPUT_ROOT
                        / variant_name
                        / f"run_{variant_name}.log"
                    ),
                    "naive_mrr": None,
                    "naive_mr": None,
                    "naive_hits@1": None,
                    "naive_hits@3": None,
                    "naive_hits@10": None,
                    "type_mrr": None,
                    "type_mr": None,
                    "type_hits@1": None,
                    "type_hits@3": None,
                    "type_hits@10": None,
                }
            )

    summary_df = pd.DataFrame(
        rows
    )

    summary_csv_path = (
        OUTPUT_ROOT
        / "summary.csv"
    )

    summary_json_path = (
        OUTPUT_ROOT
        / "summary.json"
    )

    summary_df.to_csv(
        summary_csv_path,
        index=False,
    )

    with open(
        summary_json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "results": rows,
                "failures": failures,
            },
            f,
            indent=2,
        )

    root_logger.info(
        "=" * 90
    )

    root_logger.info(
        f"Done. Success: "
        f"{sum(r.get('status') == 'ok' for r in rows)}  "
        f"Failures: {len(failures)}"
    )

    root_logger.info(
        f"Summary CSV : "
        f"{summary_csv_path}"
    )

    root_logger.info(
        f"Summary JSON: "
        f"{summary_json_path}"
    )

    root_logger.info(
        "=" * 90
    )


if __name__ == "__main__":
    main()
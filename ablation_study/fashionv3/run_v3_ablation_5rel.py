"""
run_v3_ablation_5rel.py — TransE ablation on Fashion v3, 5-relation family.

Trains TransE on each variant listed in ablation_manifest_5rel.json,
using the fixed hyperparameters selected through HPO for the 5-relation
family (embedding_dim=128).

A fixed vocabulary (entity_to_id, relation_to_id) is loaded from
entity2id.tsv and relation2id.tsv under data/processed_v3.
The vocabulary is NOT inferred from the filtered KG files, ensuring that
num_entities and num_relations remain identical across all variants.

Usage:
    python run_v3_ablation_5rel.py
    python run_v3_ablation_5rel.py --variant loo_no_belongs_to
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import pandas as pd
import torch

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE = Path(__file__).resolve().parents[2]

FASHION_BASE = BASE / "fashion_generalization"
DATA_DIR = FASHION_BASE / "data" / "processed_v3"

ABLATION_DIR = BASE / "ablation_study" / "fashionv3" / "ablation_5rel"
MANIFEST_PATH = ABLATION_DIR / "ablation_manifest_5rel.json"

VALID_PATH = DATA_DIR / "taskA_valid.tsv"
TEST_PATH = DATA_DIR / "taskA_test.tsv"
ENTITY2ID_PATH = DATA_DIR / "entity2id.tsv"
RELATION2ID_PATH = DATA_DIR / "relation2id.tsv"

OUTPUT_ROOT = BASE / "ablation_study" / "fashionv3" / "results_5rel"


RANDOM_SEED = 42
TRAINING_EPOCHS_FINAL = 300
BATCH_SIZE = 1024
RELATION_LABEL = "compatible_with"
HITS_AT_K = [1, 3, 10]


# Fixed best parameters from TransE HPO on the Fashion v3 5-relation family
BEST_PARAMS = {
    "embedding_dim": 128,
    "learning_rate": 0.0003048761416961341,
    "num_negs_per_pos": 8,
    "loss": "marginranking",
    "loss_margin": 9.671773805079942,
}


EARLY_STOPPING_KWARGS = {
    "frequency": 5,
    "patience": 3,
    "metric": "inverse_harmonic_mean_rank",
    "relative_delta": 0.002,
}


TYPE_PREFIXES = [
    "item",
    "category",
    "brand",
    "pricetier",
    "poptier",
]


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def setup_root_logger() -> logging.Logger:
    logger = logging.getLogger("ablation_v3_5rel")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(
            logging.Formatter("%(asctime)s - %(message)s")
        )
        logger.addHandler(sh)

    return logger


def setup_variant_logger(
    variant_name: str,
    variant_dir: Path,
) -> logging.Logger:

    logger = logging.getLogger(
        f"ablation_v3_5rel.{variant_name}"
    )
    logger.setLevel(logging.INFO)
    logger.propagate = True

    for h in list(logger.handlers):
        logger.removeHandler(h)

    fh = logging.FileHandler(
        variant_dir / f"run_{variant_name}.log",
        mode="w",
        encoding="utf-8",
    )

    fh.setFormatter(
        logging.Formatter("%(asctime)s - %(message)s")
    )

    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_manifest(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_fixed_vocab(logger: logging.Logger):

    e2id_df = pd.read_csv(
        ENTITY2ID_PATH,
        sep="\t"
    )

    r2id_df = pd.read_csv(
        RELATION2ID_PATH,
        sep="\t"
    )

    entity_to_id = dict(
        zip(
            e2id_df["entity"],
            e2id_df["id"]
        )
    )

    relation_to_id = dict(
        zip(
            r2id_df["relation"],
            r2id_df["id"]
        )
    )

    logger.info(
        f"Fixed vocabulary: "
        f"{len(entity_to_id):,} entities, "
        f"{len(relation_to_id)} relations"
    )

    return entity_to_id, relation_to_id


def load_triples_factories(
    kg_train_path,
    entity_to_id,
    relation_to_id,
    logger,
):

    train_df = pd.read_csv(
        kg_train_path,
        sep="\t"
    )

    valid_df = pd.read_csv(
        VALID_PATH,
        sep="\t"
    )

    test_df = pd.read_csv(
        TEST_PATH,
        sep="\t"
    )

    train_arr = train_df[
        ["head", "relation", "tail"]
    ].values

    valid_arr = valid_df[
        ["head", "relation", "tail"]
    ].values

    test_arr = test_df[
        ["head", "relation", "tail"]
    ].values

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

    logger.info(
        f"  Entities : "
        f"{train_factory.num_entities:,}"
    )

    logger.info(
        f"  Relations: "
        f"{train_factory.num_relations:,}"
    )

    logger.info(
        f"  Triples  : "
        f"train={train_factory.num_triples:,} (filtered), "
        f"valid={valid_factory.num_triples:,}, "
        f"test={test_factory.num_triples:,}"
    )

    return (
        train_factory,
        valid_factory,
        test_factory,
    )


def build_type_index(entity_to_id):

    type_to_ids = {
        p: []
        for p in TYPE_PREFIXES
    }

    for label, eid in entity_to_id.items():

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

    device = next(
        model.parameters()
    ).device

    relation_id = (
        train_factory
        .relation_to_id[
            relation_label
        ]
    )

    num_entities = (
        train_factory.num_entities
    )

    # compatible_with is an item-to-item relation:
    # use the same type constraint for heads and tails.
    if mode == "type_constrained":

        item_ids = (
            type_index["item"]
            .to(device)
        )

        tail_mask = torch.full(
            (num_entities,),
            float("-inf"),
            device=device,
        )

        tail_mask[item_ids] = 0.0

        head_mask = torch.full(
            (num_entities,),
            float("-inf"),
            device=device,
        )

        head_mask[item_ids] = 0.0

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

        for h, r, t in (
            fac.mapped_triples.tolist()
        ):

            if r != relation_id:
                continue

            head_to_known_tails.setdefault(
                h,
                set()
            ).add(t)

            tail_to_known_heads.setdefault(
                t,
                set()
            ).add(h)

    test_triples = (
        test_factory.mapped_triples
    )

    test_triples = test_triples[
        test_triples[:, 1]
        == relation_id
    ]

    model.eval()

    tail_ranks = []
    head_ranks = []

    with torch.no_grad():

        for h, r, t in (
            test_triples.tolist()
        ):

            scores = (
                model.score_t(
                    torch.tensor(
                        [[h, r]],
                        device=device,
                    )
                ).squeeze(0)
                + tail_mask
            )

            for kt in (
                head_to_known_tails
                .get(h, set())
            ):

                if kt != t:
                    scores[kt] = float("-inf")

            tail_ranks.append(
                (
                    scores > scores[t]
                ).sum().item()
                + 1
            )

            scores = (
                model.score_h(
                    torch.tensor(
                        [[r, t]],
                        device=device,
                    )
                ).squeeze(0)
                + head_mask
            )

            for kh in (
                tail_to_known_heads
                .get(t, set())
            ):

                if kh != h:
                    scores[kh] = float("-inf")

            head_ranks.append(
                (
                    scores > scores[h]
                ).sum().item()
                + 1
            )

    def metrics(ranks):

        ranks = torch.tensor(
            ranks
        ).float()

        m = {
            "MRR": (
                1.0 / ranks
            ).mean().item(),
            "MR": ranks.mean().item(),
        }

        for k in HITS_AT_K:

            m[f"Hits@{k}"] = (
                ranks <= k
            ).float().mean().item()

        return m

    tail_m = metrics(
        tail_ranks
    )

    head_m = metrics(
        head_ranks
    )

    avg_m = {
        k: (
            tail_m[k] + head_m[k]
        ) / 2
        for k in tail_m
    }

    return {
        "tail": tail_m,
        "head": head_m,
        "avg": avg_m,
    }


def build_pipeline_kwargs(
    train_factory,
    valid_factory,
    device,
    best_params,
):

    return dict(
        training=train_factory,
        validation=valid_factory,
        testing=valid_factory,
        model="TransE",
        model_kwargs={
            "embedding_dim":
                best_params[
                    "embedding_dim"
                ]
        },
        loss=best_params["loss"],
        loss_kwargs={
            "margin":
                best_params[
                    "loss_margin"
                ]
        },
        optimizer="Adam",
        optimizer_kwargs={
            "lr":
                best_params[
                    "learning_rate"
                ]
        },
        training_kwargs={
            "num_epochs":
                TRAINING_EPOCHS_FINAL,
            "batch_size":
                BATCH_SIZE,
        },
        negative_sampler="basic",
        negative_sampler_kwargs={
            "num_negs_per_pos":
                best_params[
                    "num_negs_per_pos"
                ]
        },
        stopper="early",
        stopper_kwargs=(
            EARLY_STOPPING_KWARGS
        ),
        random_seed=RANDOM_SEED,
        device=device,
        use_tqdm=False,
    )


def flatten_metrics(
    prefix,
    results,
):

    avg = results["avg"]

    return {
        f"{prefix}_mrr":
            avg["MRR"],
        f"{prefix}_mr":
            avg["MR"],
        f"{prefix}_hits@1":
            avg["Hits@1"],
        f"{prefix}_hits@3":
            avg["Hits@3"],
        f"{prefix}_hits@10":
            avg["Hits@10"],
    }


# ---------------------------------------------------------------------
# Run a single variant
# ---------------------------------------------------------------------

def run_one_variant(
    variant_name,
    spec,
    entity_to_id,
    relation_to_id,
    root_logger,
):

    variant_dir = (
        OUTPUT_ROOT
        / variant_name
    )

    variant_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    variant_logger = (
        setup_variant_logger(
            variant_name,
            variant_dir,
        )
    )

    variant_logger.info(
        f"Variant: {variant_name} "
        f"(kind={spec.get('kind')})"
    )

    variant_logger.info(
        f"Relations kept: "
        f"{spec.get('relations_kept')}"
    )

    kg_train_path = (
        ABLATION_DIR
        / f"kg_train_{variant_name}.tsv"
    )

    (
        train_factory,
        valid_factory,
        test_factory,
    ) = load_triples_factories(
        kg_train_path,
        entity_to_id,
        relation_to_id,
        logger=variant_logger,
    )

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
        f"num_entities="
        f"{train_factory.num_entities}, "
        f"num_relations="
        f"{train_factory.num_relations} "
        f"(fixed across all variants)"
    )

    type_index = (
        build_type_index(
            entity_to_id
        )
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

    for k, v in (
        BEST_PARAMS.items()
    ):

        variant_logger.info(
            f"  {k:18s} = {v}"
        )

    t0 = time.time()

    result = pipeline(
        **build_pipeline_kwargs(
            train_factory,
            valid_factory,
            device,
            BEST_PARAMS,
        )
    )

    elapsed = (
        time.time() - t0
    )

    variant_logger.info(
        f"Training completed "
        f"in {elapsed:.0f}s"
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
                "variant":
                    variant_name,
                "kind":
                    spec.get("kind"),
                "kg_train_ablation":
                    str(kg_train_path),
                "naive":
                    naive_results,
                "type_constrained":
                    typed_results,
                "elapsed_seconds":
                    elapsed,
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
                "variant":
                    variant_name,
                "kind":
                    spec.get("kind"),
                "kg_train_ablation":
                    str(kg_train_path),
                "relations_kept":
                    spec.get(
                        "relations_kept",
                        [],
                    ),
                "relations_excluded":
                    spec.get(
                        "relations_excluded",
                        [],
                    ),
                "best_params":
                    BEST_PARAMS,
                "random_seed":
                    RANDOM_SEED,
                "training_epochs_final":
                    TRAINING_EPOCHS_FINAL,
                "batch_size":
                    BATCH_SIZE,
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
        (
            "naive",
            naive_results,
        ),
        (
            "type_constrained",
            typed_results,
        ),
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

    row = {
        "variant":
            variant_name,
        "kind":
            spec.get(
                "kind",
                ""
            ),
        "kg_train_ablation":
            str(kg_train_path),
        "status":
            "ok",
        "elapsed_seconds":
            round(
                elapsed,
                4
            ),
        "best_model_path":
            str(best_model_path),
        "best_metrics_path":
            str(best_metrics_path),
        "log_path":
            str(
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

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help=(
            "Name of a single variant to run. "
            "If omitted, all variants are executed."
        ),
    )

    args = parser.parse_args()

    root_logger = (
        setup_root_logger()
    )

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
                f"not found. Available variants: "
                f"{list(manifest.keys())}"
            )

        manifest = {
            args.variant:
                manifest[
                    args.variant
                ]
        }

        root_logger.info(
            f"Single-variant mode: "
            f"{args.variant}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        entity_to_id,
        relation_to_id,
    ) = load_fixed_vocab(
        root_logger
    )

    root_logger.info(
        f"Manifest path: "
        f"{MANIFEST_PATH}"
    )

    root_logger.info(
        f"Output root  : "
        f"{OUTPUT_ROOT}"
    )

    root_logger.info(
        f"Variants     : "
        f"{len(manifest)}"
    )

    rows = []
    failures = {}

    for (
        variant_name,
        spec,
    ) in manifest.items():

        variant_dir = (
            OUTPUT_ROOT
            / variant_name
        )

        if (
            variant_dir
            / f"best_metrics_{variant_name}.json"
        ).exists():

            root_logger.info(
                f"Skipping already "
                f"completed variant: "
                f"{variant_name}"
            )

            continue

        root_logger.info(
            f"Starting variant: "
            f"{variant_name}"
        )

        try:

            row = run_one_variant(
                variant_name,
                spec,
                entity_to_id,
                relation_to_id,
                root_logger,
            )

            rows.append(row)

        except Exception as exc:

            failures[
                variant_name
            ] = str(exc)

            root_logger.exception(
                f"Variant failed: "
                f"{variant_name} "
                f"-> {exc}"
            )

            rows.append(
                {
                    "variant":
                        variant_name,
                    "kind":
                        spec.get(
                            "kind",
                            ""
                        ),
                    "kg_train_ablation":
                        str(
                            spec.get(
                                "kg_train_ablation",
                                ""
                            )
                        ),
                    "status":
                        "failed",
                    "error":
                        str(exc),
                }
            )

    summary_df = pd.DataFrame(
        rows
    )

    summary_df.to_csv(
        OUTPUT_ROOT / "summary.csv",
        index=False,
    )

    with open(
        OUTPUT_ROOT / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "results":
                    rows,
                "failures":
                    failures,
            },
            f,
            indent=2,
        )

    root_logger.info(
        "=" * 90
    )

    root_logger.info(
        f"Done. "
        f"Success: "
        f"{sum(r.get('status') == 'ok' for r in rows)}  "
        f"Failures: "
        f"{len(failures)}"
    )

    root_logger.info(
        "=" * 90
    )


if __name__ == "__main__":
    main()
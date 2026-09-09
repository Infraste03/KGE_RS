"""
generate_ablation_plan.py

Generates the B2B ablation plan (leave-one-out + random mix)
and creates the corresponding filtered KG training files.

Usage:
    python generate_ablation_plan.py \
        --kg-train data/processed/kg_train.tsv \
        --output-dir data/processed/ablation \
        --manifest data/processed/ablation/ablation_manifest.json \
        --seed 42
"""

import argparse
import json
import os
import random

import pandas as pd


RELATIONS = [
    "bought",
    "compatible_with",
    "compatible_with_model",
    "instance_of",
    "owns",
]


def filter_kg(input_path, output_path, keep_relations):
    df = pd.read_csv(input_path, sep="\t")

    assert {
        "head",
        "relation",
        "tail",
    }.issubset(df.columns)

    n_before = len(df)

    df_filtered = df[
        df["relation"].isin(keep_relations)
    ]

    n_after = len(df_filtered)

    df_filtered.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    print(
        f"  {os.path.basename(output_path):45s} "
        f"keep={sorted(keep_relations)}  "
        f"triples {n_before:,} -> {n_after:,}"
    )


def build_loo_variants():
    variants = {}

    for rel in RELATIONS:
        name = f"loo_no_{rel}"

        keep = [
            r
            for r in RELATIONS
            if r != rel
        ]

        variants[name] = keep

    return variants


def build_random_mix_variants(
    n_combos,
    rng,
    existing_sets,
):
    variants = {}

    existing_sets = [
        frozenset(s)
        for s in existing_sets
    ]

    attempts = 0

    while (
        len(variants) < n_combos
        and attempts < 1000
    ):
        attempts += 1

        size = rng.choice(
            [2, 3, 4]
        )

        combo = frozenset(
            rng.sample(
                RELATIONS,
                size,
            )
        )

        if (
            combo in existing_sets
            or combo
            in [
                frozenset(v)
                for v in variants.values()
            ]
        ):
            continue

        idx = len(variants) + 1

        name = (
            f"mix{idx}_keep_"
            f"{'_'.join(sorted(combo))}"
        )

        variants[name] = sorted(combo)

        existing_sets.append(combo)

    if len(variants) < n_combos:
        raise RuntimeError(
            "Unable to generate enough unique combinations. "
            "Run again with a different random seed."
        )

    return variants


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--kg-train",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--n-random-mix",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    rng = random.Random(
        args.seed
    )

    loo_variants = build_loo_variants()

    mix_variants = build_random_mix_variants(
        args.n_random_mix,
        rng,
        existing_sets=list(
            loo_variants.values()
        ),
    )

    all_variants = {
        **loo_variants,
        **mix_variants,
    }

    print("=" * 70)

    print(
        f"ABLATION PLAN: "
        f"{len(loo_variants)} leave-one-out + "
        f"{len(mix_variants)} random mix"
    )

    print("=" * 70)

    manifest = {}

    for name, keep_relations in all_variants.items():
        out_path = os.path.join(
            args.output_dir,
            f"kg_train_{name}.tsv",
        )

        filter_kg(
            args.kg_train,
            out_path,
            keep_relations,
        )

        manifest[name] = {
            "kg_train_ablation": out_path,
            "relations_kept": sorted(
                keep_relations
            ),
            "relations_excluded": sorted(
                set(RELATIONS)
                - set(keep_relations)
            ),
            "kind": (
                "loo"
                if name in loo_variants
                else "random_mix"
            ),
        }

    with open(
        args.manifest,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print(
        f"\nManifest saved to: "
        f"{args.manifest}"
    )

    print(
        f"Total generated variants: "
        f"{len(manifest)}"
    )


if __name__ == "__main__":
    main()
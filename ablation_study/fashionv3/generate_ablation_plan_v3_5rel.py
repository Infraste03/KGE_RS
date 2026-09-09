"""
generate_ablation_plan_v3_5rel.py — Generate the ablation plan for Fashion v3,
5-relation family (compatible_with, belongs_to, belongs_to_brand,
belongs_to_price_tier, belongs_to_pop_tier).

Structural counterpart of generate_ablation_plan.py and
generate_ablation_plan_extra.py for the B2B domain.

The default configuration generates:
- 5 leave-one-out variants
- 3 random-mix variants
- 5 targeted variants centered on compatible_with

for a total of 13 variants.

Usage:
    python generate_ablation_plan_v3_5rel.py \
        --kg-train ../../fashion_generalization/data/processed_v3/kg_train.tsv \
        --output-dir ablation_5rel \
        --manifest ablation_5rel/ablation_manifest_5rel.json \
        --seed 42
"""

import argparse
import json
import os
import random

import pandas as pd


RELATIONS = [
    "compatible_with",
    "belongs_to",
    "belongs_to_brand",
    "belongs_to_price_tier",
    "belongs_to_pop_tier",
]


# Manually defined targeted variants centered on compatible_with.
# These are the Fashion counterpart of the targeted B2B ablations.
TARGETED_VARIANTS = {
    "targeted_pair_cw_brand": [
        "compatible_with",
        "belongs_to_brand",
    ],
    "targeted_pair_cw_category": [
        "compatible_with",
        "belongs_to",
    ],
    "targeted_single_cw": [
        "compatible_with",
    ],
    "targeted_triple_cw_brand_pricetier": [
        "compatible_with",
        "belongs_to_brand",
        "belongs_to_price_tier",
    ],
    "targeted_triple_cw_category_poptier": [
        "compatible_with",
        "belongs_to",
        "belongs_to_pop_tier",
    ],
}


def filter_kg(input_path, output_path, keep_relations):
    df = pd.read_csv(
        input_path,
        sep="\t",
    )

    assert {
        "head",
        "relation",
        "tail",
    }.issubset(df.columns)

    n_before = len(df)

    df_filtered = df[
        df["relation"].isin(
            keep_relations
        )
    ]

    n_after = len(df_filtered)

    df_filtered.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    print(
        f"  {os.path.basename(output_path):55s} "
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
            or combo in [
                frozenset(v)
                for v in variants.values()
            ]
        ):
            continue

        idx = (
            len(variants) + 1
        )

        # Short relation names are used to keep variant names readable.
        short_names = {
            "compatible_with": "cw",
            "belongs_to": "cat",
            "belongs_to_brand": "brand",
            "belongs_to_price_tier": "price",
            "belongs_to_pop_tier": "pop",
        }

        name = (
            f"mix{idx}_keep_"
            f"{'_'.join(short_names[r] for r in sorted(combo))}"
        )

        variants[name] = sorted(
            combo
        )

        existing_sets.append(
            combo
        )

    if len(variants) < n_combos:
        raise RuntimeError(
            "Unable to generate the requested number of unique "
            "relation combinations. Try using a different seed."
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

    loo_variants = (
        build_loo_variants()
    )

    targeted_variants = (
        TARGETED_VARIANTS
    )

    mix_variants = (
        build_random_mix_variants(
            args.n_random_mix,
            rng,
            existing_sets=(
                list(
                    loo_variants.values()
                )
                + list(
                    targeted_variants.values()
                )
            ),
        )
    )

    all_variants = {
        **loo_variants,
        **mix_variants,
        **targeted_variants,
    }

    print("=" * 70)

    print(
        "FASHION v3 ABLATION PLAN — 5rel"
    )

    print(
        f"  {len(loo_variants)} leave-one-out "
        f"+ {len(mix_variants)} random-mix "
        f"+ {len(targeted_variants)} targeted "
        f"= {len(all_variants)} total variants"
    )

    print("=" * 70)

    manifest = {}

    for (
        name,
        keep_relations,
    ) in all_variants.items():

        out_path = os.path.join(
            args.output_dir,
            f"kg_train_{name}.tsv",
        )

        filter_kg(
            args.kg_train,
            out_path,
            keep_relations,
        )

        if name in loo_variants:
            kind = "loo"

        elif name in mix_variants:
            kind = "random_mix"

        else:
            kind = "targeted"

        manifest[name] = {
            "kg_train_ablation":
                out_path,
            "relations_kept":
                sorted(
                    keep_relations
                ),
            "relations_excluded":
                sorted(
                    set(RELATIONS)
                    - set(keep_relations)
                ),
            "kind":
                kind,
        }

    with open(
        args.manifest,
        "w",
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
        f"Total variants generated: "
        f"{len(manifest)}"
    )


if __name__ == "__main__":
    main()
"""
generate_ablation_plan_v3_3rel.py — Generate the ablation plan for Fashion v3,
3-relation family (compatible_with, belongs_to, belongs_to_brand).

Four variants are generated:
- 3 leave-one-out variants
- 1 targeted variant containing only compatible_with

No random-mix variants are generated because, with only three relations,
the combinatorial space is too small for them to provide meaningful
additional configurations beyond the leave-one-out and targeted variants.

Usage:
    python generate_ablation_plan_v3_3rel.py \
        --kg-train ../../fashion_generalization/data/processed_v3/kg_train.tsv \
        --output-dir ablation_3rel \
        --manifest ablation_3rel/ablation_manifest_3rel.json
"""

import argparse
import json
import os

import pandas as pd


RELATIONS = [
    "compatible_with",
    "belongs_to",
    "belongs_to_brand",
]


TARGETED_VARIANTS = {
    "targeted_single_cw": ["compatible_with"],
}


def filter_kg(input_path, output_path, keep_relations):
    df = pd.read_csv(input_path, sep="\t")

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

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    loo_variants = (
        build_loo_variants()
    )

    targeted_variants = (
        TARGETED_VARIANTS
    )

    all_variants = {
        **loo_variants,
        **targeted_variants,
    }

    print("=" * 70)

    print(
        "FASHION v3 ABLATION PLAN — 3rel"
    )

    print(
        f"  {len(loo_variants)} leave-one-out "
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

        kind = (
            "loo"
            if name in loo_variants
            else "targeted"
        )

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
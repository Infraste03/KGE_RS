"""
generate_ablation_plan_extra.py

Extends the B2B ablation plan with targeted combinations centered on
'compatible_with', without modifying the already existing variants
(loo_*, mix1/2/3_*).

Usage:
    python generate_ablation_plan_extra.py \
        --kg-train data/processed/kg_train.tsv \
        --output-dir data/processed/ablation \
        --manifest data/processed/ablation/ablation_manifest.json
"""

import argparse
import json
import os

import pandas as pd


RELATIONS = [
    "bought",
    "compatible_with",
    "compatible_with_model",
    "instance_of",
    "owns",
]


# Manually defined targeted combinations.
# All configurations are centered on 'compatible_with'.
TARGETED_VARIANTS = {
    "targeted_pair_cw_cwm": [
        "compatible_with",
        "compatible_with_model",
    ],
    "targeted_pair_cw_instance": [
        "compatible_with",
        "instance_of",
    ],
    "targeted_single_cw": [
        "compatible_with",
    ],
    "targeted_triple_cw_cwm_owns": [
        "compatible_with",
        "compatible_with_model",
        "owns",
    ],
    "targeted_triple_cw_bought_instance": [
        "compatible_with",
        "bought",
        "instance_of",
    ],
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

    # Load the existing manifest, if available, so that new variants
    # can be added without overwriting the existing ones.
    if os.path.exists(args.manifest):

        with open(
            args.manifest,
            "r",
            encoding="utf-8",
        ) as f:
            manifest = json.load(f)

        print(
            f"Existing manifest loaded: "
            f"{len(manifest)} variants already present"
        )

    else:
        manifest = {}

        print(
            "No existing manifest found. "
            "A new one will be created."
        )

    print("=" * 70)
    print(
        f"ADDITIONAL TARGETED VARIANTS: "
        f"{len(TARGETED_VARIANTS)}"
    )
    print("=" * 70)

    for name, keep_relations in TARGETED_VARIANTS.items():

        if name in manifest:
            print(
                f"  SKIP "
                f"(already present in manifest): "
                f"{name}"
            )
            continue

        out_path = os.path.join(
            args.output_dir,
            f"kg_train_{name}.tsv",
        )

        if os.path.exists(out_path):
            print(
                f"  WARNING: {out_path} already exists on disk. "
                f"Skipping to avoid overwriting it. "
                f"Delete it manually if you want to regenerate it."
            )
            continue

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
            "kind": "targeted",
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
        f"\nUpdated manifest saved to: "
        f"{args.manifest}"
    )

    print(
        f"Total variants in manifest: "
        f"{len(manifest)}"
    )


if __name__ == "__main__":
    main()
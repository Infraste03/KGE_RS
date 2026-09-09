"""
collect_transe_ablation.py
==========================

Reads all best_metrics_<variant>.json files from the subdirectories of
ablation_transe/ and extracts MRR and Hits@10 from the type-constrained
average evaluation.

Usage:
    python collect_transe_ablation.py
"""

import json
from pathlib import Path


BASE_DIR = Path(__file__).parent

rows = []


for variant_dir in sorted(BASE_DIR.iterdir()):
    if not variant_dir.is_dir():
        continue

    json_files = list(
        variant_dir.glob("best_metrics_*.json")
    )

    if not json_files:
        continue

    json_path = json_files[0]

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    variant = data.get(
        "variant",
        variant_dir.name,
    )

    kind = data.get(
        "kind",
        "",
    )

    try:
        mrr = data[
            "type_constrained"
        ]["avg"]["MRR"]

        hits10 = data[
            "type_constrained"
        ]["avg"]["Hits@10"]

    except KeyError:
        print(
            f"[WARNING] Missing fields in {json_path}"
        )
        continue

    rows.append(
        (
            variant,
            kind,
            mrr,
            hits10,
        )
    )


# Print a readable summary table.
print(
    f"{'Variant':<45} "
    f"{'Kind':<10} "
    f"{'MRR':>10} "
    f"{'Hits@10':>10}"
)

print("-" * 78)

for variant, kind, mrr, hits10 in rows:
    print(
        f"{variant:<45} "
        f"{kind:<10} "
        f"{mrr:>10.4f} "
        f"{hits10:>10.4f}"
    )


# Print LaTeX-ready rows for direct use in tables or slides.
print(
    "\n--- LaTeX rows ---\n"
)

for variant, kind, mrr, hits10 in rows:
    variant_tex = variant.replace(
        "_",
        "\\_",
    )

    print(
        f"{variant_tex} & "
        f"{mrr:.4f} & "
        f"{hits10:.4f} \\\\"
    )
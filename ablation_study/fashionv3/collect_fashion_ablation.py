"""
collect_fashion_ablation.py
=============================

Reads all best_metrics_<variant>.json files from the subdirectories of
results_3rel/ and results_5rel/ and extracts MRR and Hits@10.

Usage:
    python collect_fashion_ablation.py
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
SUBDIRS = ["results_3rel", "results_5rel"]


def extract_mrr_hits10(data):
    """
    First try the B2B-style structure (type_constrained.avg),
    then fall back to direct top-level metric keys.
    """
    try:
        return (
            data["type_constrained"]["avg"]["MRR"],
            data["type_constrained"]["avg"]["Hits@10"],
        )
    except (KeyError, TypeError):
        pass

    # Fallback: direct top-level keys
    for mrr_key in ("MRR", "mrr"):
        for hits_key in ("Hits@10", "H@10", "hits@10", "hits_at_10"):
            if mrr_key in data and hits_key in data:
                return data[mrr_key], data[hits_key]

    return None, None


def main():
    rows = []
    unmatched = []

    for subdir in SUBDIRS:
        rel_dir = BASE_DIR / subdir

        if not rel_dir.exists():
            print(f"[SKIP] {rel_dir} not found")
            continue

        for variant_dir in sorted(rel_dir.iterdir()):
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
                encoding="utf-8"
            ) as f:
                data = json.load(f)

            variant = data.get(
                "variant",
                variant_dir.name
            )

            kind = data.get(
                "kind",
                ""
            )

            mrr, hits10 = extract_mrr_hits10(
                data
            )

            if mrr is None:
                unmatched.append(
                    (
                        json_path,
                        list(data.keys())
                    )
                )
                continue

            rows.append(
                (
                    subdir,
                    variant,
                    kind,
                    mrr,
                    hits10
                )
            )

    # Human-readable table
    print(
        f"{'RelSet':<14} "
        f"{'Variant':<42} "
        f"{'Kind':<12} "
        f"{'MRR':>10} "
        f"{'Hits@10':>10}"
    )

    print("-" * 92)

    for (
        relset,
        variant,
        kind,
        mrr,
        hits10
    ) in rows:

        print(
            f"{relset:<14} "
            f"{variant:<42} "
            f"{kind:<12} "
            f"{mrr:>10.4f} "
            f"{hits10:>10.4f}"
        )

    print(
        "\n--- LaTeX Rows ---\n"
    )

    for (
        relset,
        variant,
        kind,
        mrr,
        hits10
    ) in rows:

        variant_tex = variant.replace(
            "_",
            "\\_"
        )

        print(
            f"{variant_tex} & "
            f"{kind} & "
            f"{mrr:.4f} & "
            f"{hits10:.4f} \\\\  "
            f"% {relset}"
        )

    if unmatched:

        print(
            f"\n[WARNING] "
            f"{len(unmatched)} files "
            f"without recognized MRR/Hits@10 metrics:"
        )

        for path, keys in unmatched:

            print(
                f"  {path}"
            )

            print(
                f"    available keys: "
                f"{keys}"
            )


if __name__ == "__main__":
    main()
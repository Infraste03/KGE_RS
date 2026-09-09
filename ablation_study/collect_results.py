"""
collect_results.py
==================

Collects all best_metrics.json files under
ablation_study/step4_alternate_results/
and produces a single summary CSV:

    ablation_summary.csv

Usage:
    python collect_results.py
"""

import json
import re
import csv
from pathlib import Path


BASE_DIR = Path(__file__).parent / "step4_alternate_results"
OUTPUT_CSV = Path(__file__).parent / "ablation_summary.csv"


# Search patterns for known metrics.
# Matching is case-insensitive and tolerant to different separators
# such as "@", "_", or spaces in JSON keys.
METRIC_PATTERNS = {
    "R@20": r"r(ecall)?[\s_@]*20",
    "NDCG@20": r"ndcg[\s_@]*20",
    "MRR": r"^mrr$",
    "R@10": r"r(ecall)?[\s_@]*10",
    "NDCG@10": r"ndcg[\s_@]*10",
}


# Pattern used to interpret B2B result-folder names, for example:
#
#   step4_ablation_loo_no_compatible_with_one_to_one_seed1024
#   step4_ablation_loo_no_compatible_with_one_to_one
B2B_RE = re.compile(
    r"^step4_ablation_(?P<variant>.+?)_(?P<scheduling>one_to_one)(?:_seed(?P<seed>\d+))?$"
)


# Patterns used to interpret Fashion v1 result-folder names, for example:
#
#   step4_ablation_no_brand_epoch_ce
#   step4_ablation_no_compatible_with_one_to_one_ce
#   step4_full5rel_epoch_ce
#
# The last case corresponds to the full configuration rather than an ablation.
FASHION_ABLATION_RE = re.compile(
    r"^step4_ablation_(?P<variant>.+?)_(?P<scheduling>epoch|one_to_one)_ce$"
)

FASHION_FULL_RE = re.compile(
    r"^step4_(?P<variant>full5rel)_(?P<scheduling>epoch|one_to_one)_ce$"
)


def find_metrics_in_json(data, prefix=""):
    """
    Recursively traverses a JSON object and returns a dictionary
    {original_key: value} for every numeric value found.

    This flattened representation is also useful for debugging when
    expected metric names are not recognized.
    """
    flat = {}

    if isinstance(data, dict):
        for k, v in data.items():
            full_key = f"{prefix}.{k}" if prefix else k

            if isinstance(v, (dict, list)):
                flat.update(find_metrics_in_json(v, full_key))

            elif isinstance(v, (int, float)):
                flat[full_key] = v

    elif isinstance(data, list):
        for i, item in enumerate(data):
            flat.update(
                find_metrics_in_json(
                    item,
                    f"{prefix}[{i}]"
                )
            )

    return flat


def extract_known_metrics(flat_dict):
    """
    Attempts to match known metrics from a dictionary of flattened keys.

    Returns:
        dict: {standard_metric_name: value} for each recognized metric.
    """
    result = {}

    for std_name, pattern in METRIC_PATTERNS.items():
        for key, value in flat_dict.items():

            # Compare only the final component of the flattened key.
            leaf = key.split(".")[-1]

            if re.search(pattern, leaf, re.IGNORECASE):
                result[std_name] = value
                break

    return result


def parse_folder_name(domain, folder_name):
    if domain == "b2b":
        m = B2B_RE.match(folder_name)

        if m:
            return {
                "variant": m.group("variant"),
                "scheduling": m.group("scheduling"),
                "seed": m.group("seed") or "",
                "run_type": "ablation",
            }

    elif domain == "fashion_v1":
        m = FASHION_ABLATION_RE.match(folder_name)

        if m:
            return {
                "variant": m.group("variant"),
                "scheduling": m.group("scheduling"),
                "seed": "",
                "run_type": "ablation",
            }

        m = FASHION_FULL_RE.match(folder_name)

        if m:
            return {
                "variant": m.group("variant"),
                "scheduling": m.group("scheduling"),
                "seed": "",
                "run_type": "full_config",
            }

    # Fallback for unrecognized folder-name patterns.
    # Preserve the raw folder name so that the run is not discarded.
    return {
        "variant": folder_name,
        "scheduling": "",
        "seed": "",
        "run_type": "UNKNOWN_PATTERN",
    }


def main():
    rows = []
    unmatched_files = []

    for domain_dir in sorted(BASE_DIR.iterdir()):
        if not domain_dir.is_dir():
            continue

        domain = domain_dir.name  # "b2b" or "fashion_v1"

        for run_dir in sorted(domain_dir.iterdir()):
            json_path = run_dir / "best_metrics.json"

            if not json_path.exists():
                continue

            with open(json_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)

                except json.JSONDecodeError as e:
                    print(f"[ERROR] Invalid JSON in {json_path}: {e}")
                    continue

            flat = find_metrics_in_json(data)
            metrics = extract_known_metrics(flat)

            folder_info = parse_folder_name(
                domain,
                run_dir.name,
            )

            row = {
                "domain": domain,
                "run_type": folder_info["run_type"],
                "variant": folder_info["variant"],
                "scheduling": folder_info["scheduling"],
                "seed": folder_info["seed"],
                "folder_name": run_dir.name,
                **metrics,
            }

            rows.append(row)

            if not metrics:
                unmatched_files.append(
                    (json_path, flat)
                )

    if not rows:
        print(
            "No best_metrics.json files found. "
            "Check BASE_DIR."
        )
        return

    # Fixed columns followed by the union of all detected metric columns.
    fixed_cols = [
        "domain",
        "run_type",
        "variant",
        "scheduling",
        "seed",
        "folder_name",
    ]

    metric_cols = sorted(
        set(
            key
            for row in rows
            for key in row
            if key not in fixed_cols
        )
    )

    all_cols = fixed_cols + metric_cols

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=all_cols,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(
        f"Wrote {len(rows)} rows to {OUTPUT_CSV}"
    )

    if unmatched_files:
        print(
            f"\n[WARNING] {len(unmatched_files)} files "
            f"contained no recognized metrics:"
        )

        for path, flat in unmatched_files:
            print(f"\n  {path}")
            print(
                "  Numeric keys found in JSON: "
                f"{list(flat.keys())}"
            )

        print(
            "\nUpdate METRIC_PATTERNS at the top of the script "
            "using the exact key names shown above, then run the script again."
        )


if __name__ == "__main__":
    main()
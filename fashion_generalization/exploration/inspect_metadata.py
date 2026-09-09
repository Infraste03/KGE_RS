# fashion_generalization/exploration/inspect_metadata.py

import os
import gzip
import json
from collections import Counter
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FASHION_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))

META_FILE = os.path.join(
    FASHION_ROOT,
    "data",
    "raw",
    "meta_Clothing_Shoes_and_Jewelry.json.gz"
)

SAMPLE_SIZE = 50_000  # Analyze the first 50,000 records to obtain an overview


def inspect_metadata_fields(meta_file_path, sample_size):
    """
    Analyzes the fields available in a sample of records from the metadata
    JSONL file.
    """
    field_counts = Counter()
    records_analyzed = 0

    print(f"Starting analysis of {sample_size:,} records from {meta_file_path}...")

    with gzip.open(meta_file_path, "rt", encoding="utf-8") as f:
        for line in f:
            if records_analyzed >= sample_size:
                break

            try:
                record = json.loads(line.strip())

                # Increment the count for each field present in the record
                for key in record.keys():
                    field_counts[key] += 1

                records_analyzed += 1

            except (json.JSONDecodeError, UnicodeDecodeError):
                # Ignore malformed lines
                continue

    print("Analysis completed.")

    if not field_counts:
        print("No records analyzed or no fields found.")
        return None

    # Prepare data for display
    df = pd.DataFrame.from_dict(
        field_counts,
        orient='index',
        columns=['count']
    )

    df['coverage_pct'] = (df['count'] / records_analyzed) * 100
    df = df.sort_values('count', ascending=False)

    return df


if __name__ == "__main__":
    stats_df = inspect_metadata_fields(META_FILE, SAMPLE_SIZE)

    if stats_df is not None:
        print("\nField frequency and coverage in the metadata dataset:")
        print(
            stats_df.to_string(
                formatters={'coverage_pct': '{:,.2f}%'.format}
            )
        )
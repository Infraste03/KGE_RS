# Fashion-Heavy Data

This directory contains the intermediate data produced for the Fashion-Heavy preprocessing pipeline.

It contains:

- `interactions_5_core_filtered.csv`
- `metadata_valid_items.json`
- `also_buy_pairs.json`

These files are used to construct the Fashion-Heavy knowledge graph and recommendation dataset.

The corresponding processed outputs are stored in:

```text
data/processed_v3/
data/recbole_v3/
```

This directory is kept separate from the previous preprocessing outputs to preserve
the exact inputs used for the Fashion-Heavy experiments and ensure reproducibility.
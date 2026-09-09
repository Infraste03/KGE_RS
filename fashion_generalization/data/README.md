# Data

This directory contains the main data resources used by the Fashion experiments.

It includes:

- `raw/`: original Amazon Fashion source files.
- `processed/`: processed knowledge graph for the standard 3-relation Fashion setup.
- `processed_v2/`: processed knowledge graph for the enriched v2 setup.
- `processed_v3/`: processed knowledge graph for the Fashion v3 setup.
- `recbole/`: RecBole-formatted interaction data for the standard Fashion experiments.
- `recbole_v3/`: RecBole-formatted interaction data for Fashion v3.
- `also_buy_pairs.json`, `interactions_5_core_filtered.csv`, and `metadata_valid_items.json`: intermediate data associated with the original Fashion preprocessing pipeline.

The directories `data_v2/` and `data_v3/`, located at the same level as this directory, contain the intermediate datasets used to build the v2 and v3 processed data respectively.

They are intentionally kept separate to preserve the different preprocessing versions and improve experiment reproducibility.
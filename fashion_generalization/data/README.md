# Data

This directory contains the main data resources used by the Fashion experiments.

It includes:

- `processed/`: processed knowledge graph for the 3-relation Fashion-Random setup.
- `processed_v2/`: processed knowledge graph for the enriched five-relation Fashion-Random setup.
- `processed_v3/`: processed knowledge graph for the Fashion-Heavy setup.
- `recbole/`: RecBole-formatted interaction data for Fashion-Random.
- `recbole_v3/`: RecBole-formatted interaction data for Fashion-Heavy.
- `also_buy_pairs.json`, `interactions_5_core_filtered.csv`, and `metadata_valid_items.json`: intermediate data associated with the original Fashion preprocessing pipeline.

The directories `data_v2/` and `data_v3/`, located at the same level as this directory, contain the intermediate datasets used to build the enriched Fashion-Random and Fashion-Heavy processed data, respectively.

They are intentionally kept separate to preserve the different preprocessing versions and improve experiment reproducibility.
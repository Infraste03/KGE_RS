# Fashion-Heavy Alternate Learning

This folder contains the Fashion-Heavy implementation of the alternate-learning experiments.

Fashion-Heavy uses a single unified KG vocabulary shared across the 3-relation, 5-relation, and ablation configurations. The corresponding data are stored under:

```text
fashion_generalization/data/processed_v3/
```

The unified vocabulary contains five relation types, while individual experiments may train Task A using only a subset of them.

## Files

### `load_kg_fashion_v3.py`

Loads the Fashion-Heavy Knowledge Graph and builds deterministic entity and relation mappings.

Supported entity types are:

- `item`
- `category`
- `brand`
- `pricetier`
- `poptier`

### `unified_id_space_fashion_v3.py`

Builds the shared ID space between the KG and RecBole:

```text
KG IDs <-> Unified IDs <-> RecBole internal IDs
```

RecBole data are read from:

```text
data/recbole_v3/fashion_v3/
```

### `data_loaders_fashion_v3.py`

Builds the DataLoaders used during alternate learning:

- Task A: KG triples for TransE
- Task B: sequential recommendation batches for SASRec

### `eval_utils_fashion_v3.py`

Implements evaluation for both tasks:

- Task A: filtered MRR and Hits@K for `compatible_with`
- Task B: Recall@20 and NDCG@20

Task B candidates are restricted to the Fashion-Heavy item ID range.

### `run_step4_fashion_v3.py`

Runs the full alternate-learning experiment.

It supports:

```bash
--relations 3
--relations 5
```

Both configurations use the same entity and relation vocabulary; only the Task A training triples differ.

### `run_hpo_step4_fashion_v3.py`

Runs Optuna hyperparameter optimization for Step 4.

HPO is performed on the full 3-relation configuration and the selected hyperparameters are reused for the remaining Fashion-Heavy experiments.

### `run_step4_ablation_fashion_v3.py`

Runs the selected Fashion-Heavy relation-ablation experiments.

The filtered KG files and pretrained TransE checkpoints are read from:

```text
KG_RS/ablation_study/fashionv3/
```

## Outputs

Step 4 experiment outputs are stored under:

```text
alternate_learning/hpc_step4_results_v3/
```

HPO outputs are stored under:

```text
alternate_learning/hpo_results_fashion_v3/
```

The scripts use the shared SASRec and model components located in the main `alternate_learning/` directory.
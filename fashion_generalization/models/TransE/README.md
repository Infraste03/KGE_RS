# TransE Models

This directory contains the TransE training, hyperparameter optimization, and ablation scripts used for the Fashion knowledge graph experiments.

TransE is trained for knowledge graph completion, with `compatible_with` as the main relation used for link-prediction evaluation. Evaluation is performed both over the full entity set (`naive`) and over item candidates only (`type_constrained`).

## Directory Structure

```text
models/TransE/
├── 3rel/
├── 5rel/
├── generate_ablation_kg_files.py
├── run_hpo_transe_fashion.py
├── run_hpo_transe_fashion_v2.py
├── run_hpo_transe_fashion_no_brand.py
├── run_hpo_transe_fashion_no_brand_v2.py
├── run_hpo_transe_fashion_no_category.py
├── run_hpo_transe_fashion_no_compatible.py
├── run_full_hpo_transe_fashion.py
├── run_hpo_trase_price_pop.py
├── run_v3_hpo_transe.py
├── train_transe_fashion.py
└── README.md
```

## Main Scripts

| Script | Purpose |
|---|---|
| `train_transe_fashion.py` | Basic TransE training and evaluation on the standard Fashion KG. Mainly used for preliminary tests and smoke testing. |
| `run_hpo_transe_fashion.py` | Main Optuna HPO for the original 3-relation Fashion KG. |
| `run_hpo_transe_fashion_v2.py` | Alternative HPO configuration with a modified search space and longer HPO training. |
| `run_hpo_transe_fashion_no_brand.py` | Original `no_brand` ablation. Kept as a historical version. |
| `run_hpo_transe_fashion_no_brand_v2.py` | Corrected `no_brand` ablation using the fixed vocabulary of the complete KG. |
| `run_hpo_transe_fashion_no_category.py` | Ablation excluding the category relation while preserving the complete KG vocabulary. |
| `run_hpo_transe_fashion_no_compatible.py` | Worst-case ablation excluding `compatible_with` from training while still evaluating it on the test set. |
| `generate_ablation_kg_files.py` | Generates filtered KG files for relation-ablation experiments. |
| `run_full_hpo_transe_fashion.py` | HPO on the enriched Fashion KG containing five relations. |
| `run_hpo_trase_price_pop.py` | Five-relation experiment with embedding dimension fixed to 64 for compatibility with the alternate-learning architecture. |
| `run_v3_hpo_transe.py` | HPO pipeline for Fashion v3. Supports both the `3rel` and `5rel` variants through `--variant`. |

## Fashion v3: `3rel` and `5rel`

The Fashion v3 experiments use a fixed entity and relation vocabulary loaded from `data/processed_v3/`.

Two graph variants are supported:

- `3rel`: `compatible_with`, `belongs_to`, `belongs_to_brand`
- `5rel`: the same three relations plus `belongs_to_price_tier` and `belongs_to_pop_tier`

They can be trained with:

```bash
python models/TransE/run_v3_hpo_transe.py --variant 3rel
python models/TransE/run_v3_hpo_transe.py --variant 5rel
```

The corresponding outputs are stored directly in:

```text
models/TransE/3rel/
models/TransE/5rel/
```

Each directory may contain:

```text
best_model.pt
best_params.json
best_metrics.json
trials.csv
study.db
run.log
```

These directories therefore contain experiment outputs and pretrained TransE checkpoints used by the Fashion v3 alternate-learning experiments.

## Output Locations

Most non-v3 experiments store their results under:

```text
fashion_generalization/results/
```

using a separate directory for each experiment, for example:

```text
results/hpo_transe/
results/hpo_transe_v2/
results/hpo_transe_no_brand_v2/
results/hpo_transe_no_category/
results/hpo_transe_no_compatible/
results/hpo_transe_full/
results/hpo_transe_full5rel_v2/
```

Optuna studies use SQLite storage so interrupted HPO runs can be resumed.

## Notes

The corrected ablation experiments preserve the entity and relation vocabulary of the complete KG. This is necessary to maintain consistent entity IDs and allow the resulting TransE embeddings to be used safely for warm-starting the shared embedding space in the alternate-learning architecture.

The original `run_hpo_transe_fashion_no_brand.py` is retained for reproducibility, while `run_hpo_transe_fashion_no_brand_v2.py` is the corrected version used when ID alignment must be preserved.
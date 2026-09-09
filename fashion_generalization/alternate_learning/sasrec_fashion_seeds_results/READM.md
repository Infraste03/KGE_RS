# SASRec Fashion v1 - Multi-Seed Results

This folder contains the multi-seed retraining results for the standalone
SASRec baseline on the Fashion v1 dataset.

The experiments use the hyperparameters selected during the SASRec HPO phase.
Multiple random seeds were evaluated to estimate the stability of the baseline.

## Aggregate Results

| Model | Recall@20 | NDCG@20 |
|---|---:|---:|
| SASRec baseline | 0.0966 ± 0.00023 | 0.0889 ± 0.00016 |

The reported values correspond to mean ± standard deviation across the
independent seed runs.

These results are used as the Fashion v1 standalone baseline for comparison
with the alternate-learning architectures.

Fashion v3 multi-seed results will be added once the corresponding experiments
are completed.
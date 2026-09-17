# SASRec Models

This directory contains the SASRec training and hyperparameter optimization
scripts used for the Fashion experiments.

Two dataset versions are considered:

- **Fashion-Random**, using the original Fashion preprocessing pipeline.
- **Fashion-Heavy**, using the updated dataset and preprocessing introduced in the
  third experimental version.

## Files

### `run_hpo_sasrec_fashion.py`

Hyperparameter search for the Fashion-Random SASRec baseline.

Main characteristics:

- fixed embedding dimension: `64`
- random-search HPO
- Leave-One-Out evaluation protocol
- validation and test evaluation after every epoch
- model selection based on validation `NDCG@20`
- separate `best_valid.pth` and `best_test.pth` checkpoints
- automatic resume through the experiment summary CSV

Main outputs are stored under:

```text
fashion_generalization/results/sasrec_hpo/
```

The validation-selected checkpoint used by the subsequent alternate-learning
experiments is:

```text
results/sasrec_hpo/trial_017/best_valid.pth
```

### `run_hpo_sasrec_fashion_v3.py`

Hyperparameter search for the Fashion-Heavy SASRec baseline.

It follows the same evaluation and checkpointing procedure as the v1 script,
with the main differences being:

- Fashion-Heavy RecBole dataset
- fixed embedding dimension: `128`
- dedicated Fashion-Heavy output directory

Outputs are stored under:

```text
fashion_generalization/results/sasrec_hpo_v3/
```

### `run_sasrec_fashion_smoke.py`

Lightweight local smoke test for the Fashion-Random SASRec pipeline.

It runs a single epoch on CPU with a small configuration and checks:

- dataset loading
- SASRec training and evaluation
- Recall@20 and NDCG@20 computation
- Leave-One-Out split consistency
- consistency between the RecBole dataset and the original `.inter` file

Smoke-test outputs are stored under:

```text
fashion_generalization/results/sasrec_smoke/
```

## Evaluation

The main evaluation metrics are:

- `Recall@20`
- `NDCG@20`

The experimental protocol uses chronological Leave-One-Out splitting through
RecBole.

Although test performance is tracked during the HPO runs for diagnostic
purposes, model selection is based on validation performance. The
`best_valid.pth` checkpoint should therefore be used for downstream
experiments.

## HPC Notes

The HPO scripts contain cluster-specific paths and were originally executed on
the project HPC environment. These paths may need to be adapted when running
the experiments on another system.
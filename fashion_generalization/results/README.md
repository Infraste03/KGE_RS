# Fashion Experimental Results

This directory contains the selected experimental artifacts for the Fashion generalization experiments.

Only results that are relevant to the final experimental pipelines are retained here. Intermediate trials, obsolete experiments, smoke-test outputs, and superseded checkpoints are stored separately in a local archive and are not part of the public repository.

The directory contains results for:

- TransE hyperparameter optimization on the standard 3-relation Fashion KG;
- TransE hyperparameter optimization on the enriched 5-relation KG;
- TransE models used in the KG ablation study;
- SASRec hyperparameter optimization for the standard Fashion dataset;
- SASRec hyperparameter optimization for the v3 Fashion dataset;
- auxiliary direct TransE training outputs.

## Directory Structure

The cleaned directory is organized as follows:

```text
results/
├── hpo_transe/
├── hpo_transe_full5rel_v2/
├── hpo_transe_no_brand_v2/
├── hpo_transe_no_category/
├── hpo_transe_no_compatible/
├── sasrec_hpo/
├── sasrec_hpo_v3/
└── transe/
```

## TransE Results

### `hpo_transe/`

Hyperparameter optimization of TransE on the standard Fashion Knowledge Graph.

The standard KG contains three relations:

```text
compatible_with
belongs_to
belongs_to_brand
```

This is the canonical TransE model used by the standard Fashion alternate-learning pipeline.

Main files:

```text
hpo_transe/
├── best_model.pt
├── best_params.json
├── best_metrics.json
├── trials.csv
├── study.db
└── run.log
```

File roles:

- `best_model.pt`: selected TransE checkpoint;
- `best_params.json`: hyperparameters of the selected trial;
- `best_metrics.json`: link-prediction metrics for the selected model;
- `trials.csv`: summary of the HPO trials;
- `study.db`: Optuna study database;
- `run.log`: complete HPO log.

Selected hyperparameters:

```text
embedding_dim     = 64
learning_rate     = 0.0005546863
num_negs_per_pos  = 32
loss_margin       = 8.68395
```

Type-constrained average metrics:

| Metric | Value |
|---|---:|
| MRR | 0.10568 |
| MR | 22996.29 |
| Hits@1 | 0.06910 |
| Hits@3 | 0.11835 |
| Hits@10 | 0.17554 |

Canonical checkpoint:

```text
results/hpo_transe/best_model.pt
```

## Enriched 5-Relation TransE

### `hpo_transe_full5rel_v2/`

Hyperparameter optimization of TransE on the enriched Fashion v2 Knowledge Graph.

The KG contains five relations:

```text
compatible_with
belongs_to
belongs_to_brand
belongs_to_price_tier
belongs_to_pop_tier
```

This experiment replaces an earlier 5-relation HPO in which the embedding dimension was included in the search space and the selected model used 128-dimensional embeddings.

Since the joint alternate-learning architecture requires compatibility with the 64-dimensional shared embedding space, the final 5-relation HPO fixes:

```text
embedding_dim = 64
```

Main files:

```text
hpo_transe_full5rel_v2/
├── best_model.pt
├── best_params.json
├── best_metrics.json
├── trials.csv
├── study.db
└── run.log
```

Selected hyperparameters:

```text
embedding_dim     = 64
learning_rate     = 0.0004243342
num_negs_per_pos  = 8
loss_margin       = 9.40575
```

Type-constrained average metrics:

| Metric | Value |
|---|---:|
| MRR | 0.09370 |
| MR | 17358.64 |
| Hits@1 | 0.05878 |
| Hits@3 | 0.10326 |
| Hits@10 | 0.16600 |

Canonical checkpoint:

```text
results/hpo_transe_full5rel_v2/best_model.pt
```

This is the TransE checkpoint used by the final Fashion 5-relation alternate-learning experiment.

## TransE Ablation Models

The following directories contain the TransE models used for the Fashion KG ablation experiments.

Each ablation removes one component of the standard KG and independently re-optimizes TransE.

### `hpo_transe_no_brand_v2/`

TransE HPO for the `no_brand` ablation.

The `belongs_to_brand` information is removed from the KG.

Selected hyperparameters:

```text
embedding_dim     = 64
learning_rate     = 0.0004938821
num_negs_per_pos  = 64
loss_margin       = 9.19004
```

Type-constrained average metrics:

| Metric | Value |
|---|---:|
| MRR | 0.09066 |
| MR | 28694.32 |
| Hits@1 | 0.05917 |
| Hits@3 | 0.10087 |
| Hits@10 | 0.15528 |

Canonical checkpoint:

```text
results/hpo_transe_no_brand_v2/best_model.pt
```

An earlier `hpo_transe_no_brand/` experiment was superseded by this version and is not retained in the public results directory.

### `hpo_transe_no_category/`

TransE HPO for the `no_category` ablation.

The category information is removed from the KG.

Selected hyperparameters:

```text
embedding_dim     = 64
learning_rate     = 0.0006919151
num_negs_per_pos  = 32
loss_margin       = 8.69781
```

Type-constrained average metrics:

| Metric | Value |
|---|---:|
| MRR | 0.09461 |
| MR | 30331.00 |
| Hits@1 | 0.06076 |
| Hits@3 | 0.10445 |
| Hits@10 | 0.16243 |

Canonical checkpoint:

```text
results/hpo_transe_no_category/best_model.pt
```

### `hpo_transe_no_compatible/`

TransE HPO for the `no_compatible` ablation.

The `compatible_with` relation is removed from the training KG.

Selected hyperparameters:

```text
embedding_dim     = 64
learning_rate     = 0.0005268732
num_negs_per_pos  = 64
loss_margin       = 8.28057
```

Type-constrained average metrics:

| Metric | Value |
|---|---:|
| MRR | 0.00234 |
| MR | 81581.44 |
| Hits@1 | 0.00000 |
| Hits@3 | 0.00119 |
| Hits@10 | 0.00516 |

Canonical checkpoint:

```text
results/hpo_transe_no_compatible/best_model.pt
```

The very low link-prediction performance in this configuration is expected to be informative for the ablation study, since the relation targeted by Task A is removed from the training graph.

## SASRec Results

### `sasrec_hpo/`

Contains the selected SASRec trial for the standard Fashion dataset.

The original HPO generated multiple trial directories. To avoid storing redundant checkpoints, only the selected trial is retained in the cleaned repository:

```text
sasrec_hpo/
└── trial_017/
    ├── best_valid.pth
    ├── best_test.pth
    ├── epoch_history.csv
    └── run.log
```

`trial_017` is the selected SASRec configuration.

Selected configuration:

```text
hidden_size          = 64
n_layers             = 4
n_heads              = 4
inner_size           = 256
hidden_dropout_prob  = 0.5
attn_dropout_prob    = 0.3
MAX_ITEM_LIST_LENGTH = 50
loss_type            = CE
learning_rate        = 1e-4
weight_decay         = 1e-4
```

The selected validation performance was approximately:

```text
NDCG@20   = 0.1086
```

The corresponding standard Fashion test performance was approximately:

```text
Recall@20 = 0.0964
NDCG@20   = 0.0889
```

The canonical checkpoint is:

```text
results/sasrec_hpo/trial_017/best_valid.pth
```

`best_valid.pth` must be used for downstream experiments because model selection is based on validation performance.

`best_test.pth` is retained as an experimental artifact but must not be used for model selection or as the pretrained checkpoint in the final pipeline.

All alternate-learning configurations and validation scripts referring to the standard SASRec checkpoint use the canonical path above.

## SASRec v3 Results

### `sasrec_hpo_v3/`

Contains the SASRec HPO results for the Fashion v3 dataset.

The v3 dataset uses the KG-oriented user-selection strategy defined in the preprocessing pipeline.

The selected HPO configuration is `trial_003`.

The cleaned directory retains the selected trial and the HPO summary:

```text
sasrec_hpo_v3/
├── hpo_results_sasrec_fashion_v3.csv
└── trial_003/
    ├── best_valid.pth
    ├── best_test.pth
    ├── epoch_history.csv
    └── run.log
```

Selected trial hyperparameters:

```text
n_layers             = 2
n_heads              = 8
learning_rate        = 0.0003162278
hidden_dropout_prob  = 0.3
attn_dropout_prob    = 0.1
weight_decay         = 1e-5
loss_type            = CE
neg_candidate_num    = 1
```

Validation performance:

```text
best_valid_NDCG@20   = 0.1013
best_valid_Recall@20 = 0.1203
best_valid_epoch     = 8
```

Test performance obtained from the validation-selected checkpoint:

```text
NDCG@20   = 0.0770
Recall@20 = 0.0908
```

The canonical checkpoint is:

```text
results/sasrec_hpo_v3/trial_003/best_valid.pth
```

The values associated with `best_test_epoch` in the HPO CSV are diagnostic only. Final model selection is always based on the validation metric.

## Direct TransE Training

### `transe/`

Contains outputs generated by the direct TransE training script.

This directory is separate from the HPO-selected TransE checkpoints described above.

The models used as pretrained checkpoints in the final alternate-learning experiments are the HPO-selected models stored in:

```text
results/hpo_transe/
results/hpo_transe_full5rel_v2/
results/hpo_transe_no_brand_v2/
results/hpo_transe_no_category/
results/hpo_transe_no_compatible/
```

Therefore, `transe/` should be interpreted as an auxiliary/direct-training result rather than the canonical source of pretrained TransE weights for the final experiments.

## Checkpoint Selection Policy

For reproducibility, downstream experiments use the following canonical checkpoints:

| Experiment | Checkpoint |
|---|---|
| Standard 3-rel TransE | `results/hpo_transe/best_model.pt` |
| Full 5-rel TransE | `results/hpo_transe_full5rel_v2/best_model.pt` |
| No-brand ablation | `results/hpo_transe_no_brand_v2/best_model.pt` |
| No-category ablation | `results/hpo_transe_no_category/best_model.pt` |
| No-compatible ablation | `results/hpo_transe_no_compatible/best_model.pt` |
| Standard SASRec | `results/sasrec_hpo/trial_017/best_valid.pth` |
| SASRec v3 | `results/sasrec_hpo_v3/trial_003/best_valid.pth` |

For SASRec, validation-selected checkpoints (`best_valid.pth`) are the only checkpoints used for downstream model initialization and final experimental comparisons.

## Archived Results

Several intermediate or superseded outputs were intentionally removed from the public `results/` directory and retained only in a local archive.

These include, among others:

- smoke-test outputs;
- non-selected SASRec HPO trials;
- the duplicate standalone `trial_017/` directory;
- the original 5-relation TransE HPO with 128-dimensional embeddings;
- the superseded first `no_brand` TransE experiment.

The local archive is not part of the repository and is not required to reproduce the final experiments.

This cleanup keeps the public results directory focused on the checkpoints and experiment summaries that are actually used by the final Fashion pipelines.
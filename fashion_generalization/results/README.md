# Fashion Experimental Results

This directory contains the selected experimental artifacts for the Fashion generalization experiments.

Only results that are relevant to the final experimental pipelines are retained here. Intermediate trials, obsolete experiments, smoke-test outputs, and superseded checkpoints are stored separately in a local archive and are not part of the public repository.

In the paper and in the reader-facing documentation, the two main Fashion datasets are referred to as:

- **Fashion-Random**, based on random user sampling;
- **Fashion-Heavy**, based on a user-selection strategy that prioritizes category diversity and interaction activity.

Historical identifiers such as `v2` and `v3` are retained in some directory names, file names, and experiment artifacts to preserve reproducibility and compatibility with the original experimental pipeline. The `v2` identifier refers to an enriched five-relation extension of the Fashion-Random setting, while `v3` corresponds to Fashion-Heavy.

The directory contains results for:

- TransE hyperparameter optimization on the Fashion-Random 3-relation KG;
- TransE hyperparameter optimization on the enriched 5-relation Fashion-Random KG;
- TransE models used in the KG ablation study;
- SASRec hyperparameter optimization for Fashion-Random;
- SASRec hyperparameter optimization for Fashion-Heavy;
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

Hyperparameter optimization of TransE on the Fashion-Random Knowledge Graph.

The Fashion-Random KG contains three relations:

```text
compatible_with
belongs_to
belongs_to_brand
```

This is the canonical TransE model used by the Fashion-Random alternate-learning pipeline.

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

## Enriched 5-Relation Fashion-Random TransE

### `hpo_transe_full5rel_v2/`

Hyperparameter optimization of TransE on the enriched five-relation Fashion-Random Knowledge Graph.

The KG contains five relations:

```text
compatible_with
belongs_to
belongs_to_brand
belongs_to_price_tier
belongs_to_pop_tier
```

This experiment replaces an earlier five-relation HPO in which the embedding dimension was included in the search space and the selected model used 128-dimensional embeddings.

Since this Fashion-Random alternate-learning configuration uses a 64-dimensional shared embedding space, the final five-relation HPO fixes:

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

This is the TransE checkpoint used by the enriched five-relation Fashion-Random alternate-learning experiments.

## TransE Ablation Models

The following directories contain the TransE models used for the Fashion-Random KG ablation experiments.

Each ablation removes one component of the Knowledge Graph and independently re-optimizes TransE.

### `hpo_transe_no_brand_v2/`

TransE HPO for the ablation that removes `belongs_to_brand`.

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

An earlier `hpo_transe_no_brand/` experiment was superseded by this corrected version and is not retained in the public results directory.

### `hpo_transe_no_category/`

TransE HPO for the ablation that removes `belongs_to`.

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

TransE HPO for the ablation that removes `compatible_with` from the training KG.

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

The very low link-prediction performance in this configuration is informative for the ablation study because the relation targeted by Task A is removed from the training graph.

## SASRec Results

### Fashion-Random

#### `sasrec_hpo/`

Contains the selected SASRec trial for the Fashion-Random dataset.

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
NDCG@20 = 0.1086
```

The corresponding Fashion-Random test performance was approximately:

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

All alternate-learning configurations and validation scripts referring to the Fashion-Random SASRec checkpoint use the canonical path above.

### Fashion-Heavy

#### `sasrec_hpo_v3/`

Contains the SASRec HPO results for the Fashion-Heavy dataset.

Fashion-Heavy uses the category-diversity- and interaction-activity-based user-selection strategy defined in the preprocessing pipeline.

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

The models used as pretrained checkpoints in the Fashion-Random alternate-learning experiments are the HPO-selected models stored in:

```text
results/hpo_transe/
results/hpo_transe_full5rel_v2/
results/hpo_transe_no_brand_v2/
results/hpo_transe_no_category/
results/hpo_transe_no_compatible/
```

Therefore, `transe/` should be interpreted as an auxiliary direct-training result rather than the canonical source of pretrained TransE weights for the final experiments.

## Checkpoint Selection Policy

For reproducibility, downstream experiments use the following canonical checkpoints:

| Experiment | Checkpoint |
|---|---|
| Fashion-Random 3-rel TransE | `results/hpo_transe/best_model.pt` |
| Enriched Fashion-Random 5-rel TransE | `results/hpo_transe_full5rel_v2/best_model.pt` |
| Fashion-Random without `belongs_to_brand` | `results/hpo_transe_no_brand_v2/best_model.pt` |
| Fashion-Random without `belongs_to` | `results/hpo_transe_no_category/best_model.pt` |
| Fashion-Random without `compatible_with` | `results/hpo_transe_no_compatible/best_model.pt` |
| Fashion-Random SASRec | `results/sasrec_hpo/trial_017/best_valid.pth` |
| Fashion-Heavy SASRec | `results/sasrec_hpo_v3/trial_003/best_valid.pth` |

For SASRec, validation-selected checkpoints (`best_valid.pth`) are the only checkpoints used for downstream model initialization and final experimental comparisons.

## Archived Results

Several intermediate or superseded outputs were intentionally removed from the public `results/` directory and retained only in a local archive.

These include, among others:

- smoke-test outputs;
- non-selected SASRec HPO trials;
- the duplicate standalone `trial_017/` directory;
- the original five-relation Fashion-Random TransE HPO with 128-dimensional embeddings;
- the superseded first `no_brand` TransE experiment.

The local archive is not part of the repository and is not required to reproduce the final experiments.

This cleanup keeps the public results directory focused on the checkpoints and experiment summaries that are actually used by the final Fashion pipelines.
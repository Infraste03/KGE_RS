# Step 4 Alternate-Learning Ablation Results

This directory contains the **Stage 2 ablation results** obtained after
integrating the selected Knowledge Graph variants into the joint
alternate-learning recommendation architecture.

The isolated TransE ablation experiments are stored separately. This directory
contains the downstream recommendation results obtained with:

```text
Task A = Knowledge Graph embedding with TransE
Task B = Sequential recommendation
```

The effect of removing or retaining specific Knowledge Graph relations is
therefore evaluated directly on recommendation performance.

The primary recommendation metrics are:

```text
Recall@20
NDCG@20
```

## Directory Structure

```text
step4_alternate_results/
├── README.md
├── b2b/
└── fashion_v1/
```

The two subdirectories correspond to the two domains for which Stage 2
ablation outputs are currently stored here.

## B2B Results

The directory:

```text
b2b/
```

contains the Step 4 alternate-learning runs for selected B2B Knowledge Graph
ablation variants.

The evaluated variants are:

```text
loo_no_compatible_with
loo_no_instance_of
mix1_keep_compatible_with_owns
mix3_keep_bought_compatible_with
targeted_triple_cw_bought_instance
```

The full Knowledge Graph configuration is used as the reference model in the
final analysis.

All experiments use the:

```text
one_to_one
```

scheduling strategy.

For each selected variant, the directory contains one base run and additional
runs whose random seed is explicitly encoded in the folder name.

For example:

```text
step4_ablation_loo_no_compatible_with_one_to_one/
step4_ablation_loo_no_compatible_with_one_to_one_seed42/
step4_ablation_loo_no_compatible_with_one_to_one_seed999/
step4_ablation_loo_no_compatible_with_one_to_one_seed1024/
step4_ablation_loo_no_compatible_with_one_to_one_seed2024/
```

This organization supports multi-run evaluation and the computation of
aggregate recommendation metrics.

## B2B Result Files

A representative B2B run contains:

```text
best_metrics.json
best_model.pt
checkpoint_latest.pth
last_model.pt
run.log
training_history.json
```

### `best_metrics.json`

Contains the final metrics associated with the selected model.

This is the main file used when aggregating recommendation results.

### `best_model.pt`

Contains the selected model weights for the run.

### `checkpoint_latest.pth`

Contains the most recent training checkpoint and can be used for recovery or
resume purposes.

### `last_model.pt`

Contains the model state at the final executed training epoch.

### `training_history.json`

Contains the recorded training and evaluation history.

### `run.log`

Contains the execution log for the corresponding experiment.

## B2B Final Results

The consolidated B2B ablation results are reported as mean ± standard
deviation across runs.

| Variant | Recall@20 | NDCG@20 | Interpretation |
|---|---:|---:|---|
| `full` | 0.4612 ± 0.0177 | 0.2773 ± 0.0098 | Reference |
| `loo_no_compatible_with` | 0.4247 ± 0.0173 | 0.2545 ± 0.0141 | Strongest performance drop |
| `loo_no_instance_of` | **0.4658 ± 0.0126** | **0.2774 ± 0.0072** | Slightly above full |
| `mix1_keep_compatible_with_owns` | 0.4338 ± 0.0187 | 0.2665 ± 0.0072 | Below full |
| `mix3_keep_bought_compatible_with` | 0.4612 ± 0.0139 | 0.2771 ± 0.0122 | Approximately equal to full |
| `targeted_triple_cw_bought_instance` | 0.4566 ± 0.0150 | 0.2691 ± 0.0096 | Close to full |

These results indicate that the effect of the Knowledge Graph is strongly
dependent on the relations that are retained.

In particular:

- removing `compatible_with` produces the largest degradation;
- removing `instance_of` does not reduce recommendation performance;
- retaining `bought` together with `compatible_with` is sufficient to recover
  performance very close to the complete Knowledge Graph;
- reducing the graph does not automatically improve or preserve performance:
  the specific relation composition remains important.

## Fashion v1 Results

The directory:

```text
fashion_v1/
```

contains the Stage 2 alternate-learning ablation experiments for the Fashion
v1 configuration.

Four Knowledge Graph configurations are represented:

```text
full5rel
no_brand
no_category
no_compatible_with
```

Each configuration was evaluated using two scheduling strategies:

```text
epoch
one_to_one
```

and the Task B objective is identified in the directory names as:

```text
CE
```

The resulting directories are:

```text
step4_full5rel_epoch_ce
step4_full5rel_one_to_one_ce

step4_ablation_no_brand_epoch_ce
step4_ablation_no_brand_one_to_one_ce

step4_ablation_no_category_epoch_ce
step4_ablation_no_category_one_to_one_ce

step4_ablation_no_compatible_with_epoch_ce
step4_ablation_no_compatible_with_one_to_one_ce
```

## Fashion v1 Result Files

A representative Fashion v1 run contains:

```text
best_metrics.json
best_model.pt
best_model_on_test.pt
checkpoint_latest.pth
last_model.pt
run.log
training_history.json
```

The files have the same general purpose as in the B2B experiments.

The additional:

```text
best_model_on_test.pt
```

is a test-selected diagnostic checkpoint generated by the training pipeline.

Final scientific reporting should follow the model-selection protocol defined
for the corresponding experiment rather than selecting a configuration based
on test performance.

## Fashion v1 Final Results

The Fashion v1 Stage 2 ablation experiments use Cross-Entropy for Task B and
compare the `epoch` and `one_to_one` scheduling strategies.

These runs are single-seed experiments and should therefore be interpreted as
diagnostic ablation comparisons rather than multi-seed performance estimates.

| KG Configuration | Scheduling | Recall@20 | NDCG@20 |
|---|---|---:|---:|
| Full 5-rel | `epoch` | 0.1235 | **0.1100** |
| Full 5-rel | `one_to_one` | 0.1235 | 0.1099 |
| No brand | `epoch` | 0.1162 | 0.1063 |
| No brand | `one_to_one` | 0.1165 | 0.1065 |
| No category | `epoch` | 0.1235 | 0.1075 |
| No category | `one_to_one` | 0.1234 | 0.1079 |
| No `compatible_with` | `epoch` | 0.1209 | 0.1083 |
| No `compatible_with` | `one_to_one` | 0.1209 | 0.1086 |

The full Knowledge Graph configuration achieves the strongest overall ranking
quality, with the highest NDCG@20 obtained using `epoch` scheduling.

Removing brand information produces the clearest degradation among the selected
Fashion v1 configurations.

Removing category or compatibility information also changes recommendation
performance, although the effect is less pronounced than the collapse observed
for `compatible_with` in the isolated TransE ablation.

These results highlight the distinction between isolated Knowledge Graph
embedding quality and downstream recommendation quality: a relation can be
critical for link prediction while having a more moderate effect once the
Knowledge Graph is integrated into the shared alternate-learning architecture.

## Relationship with Stage 1

The ablation study is divided into two conceptually separate stages.

### Stage 1 — Isolated TransE

Measures the effect of relation composition on Knowledge Graph embedding
quality using metrics such as:

```text
MRR
Hits@10
```

### Stage 2 — Alternate Learning

Uses the corresponding Knowledge Graph configuration in the joint
recommendation architecture and measures:

```text
Recall@20
NDCG@20
```

The files in this directory correspond exclusively to **Stage 2**.

## Result Aggregation

The individual `best_metrics.json` files can be aggregated to compare
Knowledge Graph variants and, where multiple independent runs are available,
to report:

```text
mean ± standard deviation
```

The same evaluation protocol should be used across configurations before
performing direct comparisons.

## Reproducibility

These directories preserve the model checkpoints, metrics, logs, and training
histories required to trace the final ablation results back to the individual
experimental runs.

The experiment-generation and training code is stored separately under:

```text
ablation_study/
B2B_alternate_learning/
fashion_generalization/
```

while this directory acts as the collected result archive for the downstream
alternate-learning ablation experiments.
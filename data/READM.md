# B2B Knowledge Graph Data

This directory contains the processed Knowledge Graph data used for the B2B experiments.

The files are used primarily for:

```text
Task A: TransE Knowledge Graph embedding
Step 4 alternate learning
Knowledge Graph ablation experiments
```

## Directory Structure

```text
data/
└── processed/
    ├── kg_train.tsv
    ├── taskA_valid.tsv
    ├── taskA_test.tsv
    ├── kg_stats.json
    │
    └── ablation/
        ├── ablation_manifest.json
        ├── kg_train_loo_no_bought.tsv
        ├── kg_train_loo_no_compatible_with.tsv
        ├── kg_train_loo_no_compatible_with_model.tsv
        ├── kg_train_loo_no_instance_of.tsv
        ├── kg_train_loo_no_owns.tsv
        ├── kg_train_mix1_keep_compatible_with_owns.tsv
        ├── kg_train_mix2_keep_bought_owns.tsv
        ├── kg_train_mix3_keep_bought_compatible_with.tsv
        ├── kg_train_targeted_pair_cw_cwm.tsv
        ├── kg_train_targeted_pair_cw_instance.tsv
        ├── kg_train_targeted_single_cw.tsv
        ├── kg_train_targeted_triple_cw_bought_instance.tsv
        └── kg_train_targeted_triple_cw_cwm_owns.tsv
```

## Main Knowledge Graph Files

### `kg_train.tsv`

Training split of the complete B2B Knowledge Graph.

The graph contains the five relation types used in the B2B experiments:

```text
bought
compatible_with
compatible_with_model
instance_of
owns
```

### `taskA_valid.tsv`

Validation split used for Task A Knowledge Graph embedding experiments.

### `taskA_test.tsv`

Test split used for the final Task A evaluation.

The same validation and test splits are reused across the B2B ablation variants so that relation-removal experiments remain directly comparable.

### `kg_stats.json`

Contains summary statistics for the processed B2B Knowledge Graph.

## Ablation Data

The directory:

```text
processed/ablation/
```

contains the filtered training graphs used for the B2B Knowledge Graph ablation study.

Only the training graph is modified across variants.

The validation and test sets remain:

```text
processed/taskA_valid.tsv
processed/taskA_test.tsv
```

This ensures that the effect of changing the relation composition can be evaluated under the same protocol.

## Ablation Variants

The B2B ablation study contains 13 variants.

### Leave-One-Out

```text
loo_no_bought
loo_no_compatible_with
loo_no_compatible_with_model
loo_no_instance_of
loo_no_owns
```

Each configuration removes one relation from the complete five-relation graph.

### Random-Mix

```text
mix1_keep_compatible_with_owns
mix2_keep_bought_owns
mix3_keep_bought_compatible_with
```

These configurations retain selected subsets of the available relations.

### Targeted Variants

```text
targeted_pair_cw_cwm
targeted_pair_cw_instance
targeted_single_cw
targeted_triple_cw_bought_instance
targeted_triple_cw_cwm_owns
```

These configurations are centered on `compatible_with` and are designed to measure which additional relations provide useful complementary information.

## Ablation Manifest

The file:

```text
processed/ablation/ablation_manifest.json
```

describes the 13 generated configurations.

For each variant, it records information such as:

```text
kg_train_ablation
relations_kept
relations_excluded
kind
```

The manifest is used by the B2B TransE ablation pipeline to select the appropriate filtered training graph.

## Relationship with the Ablation Study

The scripts used to generate and train these variants are documented under:

```text
ablation_study/
```

In particular:

```text
ablation_study/generate_ablation_plan.py
ablation_study/generate_ablation_plan_extra.py
ablation_study/run_transe_ablation_batch.py
```

The corresponding Stage 1 and Stage 2 ablation results are also documented in:

```text
ablation_study/README.md
```

## Reproducibility

The ablation variants preserve the same entity and relation vocabulary used by the complete B2B Knowledge Graph.

This is required to maintain consistent embedding indices across experiments and to support the subsequent warm start of the shared embedding space in the alternate-learning architecture.
# Experimental Results

This directory contains the main experimental artifacts produced by the B2B
Knowledge Graph and KGSEQ experiments.

Only the result directories retained as part of the final reproducible
repository are included here.

The directory contains:

```text
results/
├── README.md
│
├── ablation_transe/
│
├── B2B/
│   └── step4_hpo/
│       ├── epoch/
│       │   ├── selected_trial_4/
│       │   └── study/
│       └── one_to_one/
│           ├── selected_trial_1/
│           └── study/
│
├── sasrec_seeds_results/
│   ├── seed42/
│   ├── seed999/
│   ├── seed1024/
│   └── seed2024/
│
└── step2_hpo/
    ├── ComplEx/
    ├── DistMult/
    ├── RotatE/
    ├── TransE/
    └── summary.json
```

## Result Organization

The results are organized according to the main stages of the B2B experimental
pipeline.

```text
Step 2
Knowledge Graph Embedding HPO
        |
        v
results/step2_hpo/

Step 3
Standalone SASRec
        |
        v
results/sasrec_seeds_results/
+ reference seed checkpoint under SASRec/

Step 4
KGSEQ alternate-learning HPO
        |
        v
results/B2B/step4_hpo/

Knowledge Graph Ablation
        |
        v
results/ablation_transe/
```

## Step 2 - Knowledge Graph Embedding HPO

The directory:

```text
step2_hpo/
```

contains the hyperparameter optimization results for four Knowledge Graph
embedding models:

```text
TransE
RotatE
DistMult
ComplEx
```

Each model directory preserves the corresponding:

```text
best model checkpoint
best hyperparameters
best evaluation metrics
Optuna study database
trial summary
execution log
```

The exact filenames differ slightly across models because they preserve the
original experiment outputs.

### Step 2 Model Comparison

The following table reports the final type-constrained link-prediction results
for the B2B Knowledge Graph.

The evaluation focuses on the `compatible_with` relation.

| Model | MRR | Hits@10 |
|---|---:|---:|
| **RotatE** | **0.3148** | **0.537** |
| DistMult | 0.2973 | 0.510 |
| TransE | 0.2953 | 0.502 |
| ComplEx | 0.1948 | 0.369 |

RotatE achieved the strongest isolated Knowledge Graph embedding performance.

TransE was nevertheless selected for the downstream KGSEQ architecture because
its real-valued translational representation can be directly used in the shared
real-valued entity embedding space without introducing an additional
representation-conversion layer.

### `summary.json`

The file:

```text
step2_hpo/summary.json
```

provides a compact comparison of the four KGE models and records:

```text
MRR
Hits@10
best isolated KGE model
KGE model selected for downstream alternate learning
```

The complete model-specific artifacts remain available inside the corresponding
subdirectories.

## TransE Artifacts

The canonical TransE HPO output is stored in:

```text
step2_hpo/TransE/
```

and contains:

```text
best_metrics.json
best_model.pt
best_paramsTransE.json
run.log
study.db
trials.csv
```

The checkpoint:

```text
step2_hpo/TransE/best_model.pt
```

is the canonical pretrained TransE model used by the B2B KGSEQ pipeline.

## RotatE Artifacts

The directory:

```text
step2_hpo/RotatE/
```

contains:

```text
best_metricsRotatE.json
best_model.pt
best_paramsRotatE.json
runRotatE.log
study.db
trials.csv
```

RotatE obtained the strongest isolated link-prediction performance among the
four evaluated KGE models.

## DistMult Artifacts

The directory:

```text
step2_hpo/DistMult/
```

contains:

```text
best_metricsDistMul.json
best_model.pt
best_paramsDistMult.json
runDistMul.log
study.db
trials.csv
```

Some filenames preserve the historical `DistMul` naming used when the
experiments were originally executed.

## ComplEx Artifacts

The directory:

```text
step2_hpo/ComplEx/
```

contains:

```text
best_metricsComplEx.json
best_model.pt
best_paramsComplEx.json
runcOMPLEXe.log
study.db
trials.csv
```

The original filenames are retained to preserve traceability with the executed
experiments.

## Step 3 - SASRec Multi-Seed Checkpoints

The directory:

```text
sasrec_seeds_results/
```

contains the additional independent SASRec retraining runs used for the
multi-seed evaluation.

The stored seeds are:

```text
42
999
1024
2024
```

Each seed directory contains the corresponding RecBole SASRec checkpoint.

For example:

```text
sasrec_seeds_results/
├── seed42/
│   └── SASRec-<timestamp>.pth
├── seed999/
│   └── SASRec-<timestamp>.pth
├── seed1024/
│   └── SASRec-<timestamp>.pth
└── seed2024/
    └── SASRec-<timestamp>.pth
```

The four checkpoints were verified to be distinct using SHA-256 hashes.

### Reference Seed

The original SASRec run corresponding to seed:

```text
2020
```

is stored separately under the canonical SASRec model directory:

```text
SASRec/pth/
```

Therefore, the complete five-seed evaluation uses:

```text
2020
42
999
1024
2024
```

The additional seed directory under `results/` should not be interpreted as the
complete SASRec model repository; it stores only the additional retraining runs.

## Step 4 - KGSEQ Hyperparameter Optimization

The directory:

```text
B2B/step4_hpo/
```

contains the retained hyperparameter optimization results for the two main
alternate-learning scheduling strategies used by KGSEQ:

```text
epoch
one_to_one
```

The structure is:

```text
B2B/step4_hpo/
├── epoch/
│   ├── selected_trial_4/
│   └── study/
│
└── one_to_one/
    ├── selected_trial_1/
    └── study/
```

### Epoch Scheduling

The selected HPO configuration for the `epoch` scheduling strategy is stored
under:

```text
B2B/step4_hpo/epoch/selected_trial_4/
```

The directory contains:

```text
best_metrics_original.json
best_model.pt
checkpoint_epoch_21.pth
config.yaml
run.log
selected_metrics.json
training_history.json
```

The corresponding HPO study is stored under:

```text
B2B/step4_hpo/epoch/study/
```

and contains:

```text
base_config.yaml
best_test.csv
best_valid.csv
optuna_step4_epoch.db
skipped_configs.txt
trials_summary.csv
```

### One-to-One Scheduling

The selected configuration for the `one_to_one` scheduling strategy is stored
under:

```text
B2B/step4_hpo/one_to_one/selected_trial_1/
```

and contains:

```text
best_metrics_original.json
best_model.pt
checkpoint_epoch_31.pth
config.yaml
run.log
selected_metrics.json
training_history.json
```

The corresponding HPO study directory contains:

```text
base_config.yaml
best_test.csv
best_valid.csv
optuna_step4_one_to_one.db
skipped_configs.txt
trials_summary.csv
```

## HPO Model Selection

The Step 4 hyperparameter search uses validation performance for model
selection.

Test-set evaluations stored in the experiment artifacts are retained for
diagnostic and final evaluation purposes and should not be used to select
hyperparameters.

The selected trial directories preserve the checkpoints and histories required
to trace the final HPO choice back to the original experiment.

## B2B TransE Ablation Results

The directory:

```text
ablation_transe/
```

contains the isolated TransE results for the B2B relation ablation study.

The complete B2B Knowledge Graph contains five relations:

```text
bought
compatible_with
compatible_with_model
instance_of
owns
```

The ablation study contains 13 variants:

```text
5 leave-one-out
3 reduced relation subsets
5 targeted configurations
```

The technical identifiers used in the experiment artifacts are retained below
for reproducibility. Reader-facing result tables instead describe each
configuration using the relations that are removed or retained.

### Leave-One-Out Variants

```text
loo_no_bought
loo_no_compatible_with
loo_no_compatible_with_model
loo_no_instance_of
loo_no_owns
```

Their semantic meaning is:

| Internal identifier | KG configuration |
|---|---|
| `loo_no_bought` | Without `bought` |
| `loo_no_compatible_with` | Without `compatible_with` |
| `loo_no_compatible_with_model` | Without `compatible_with_model` |
| `loo_no_instance_of` | Without `instance_of` |
| `loo_no_owns` | Without `owns` |

### Reduced Relation Subsets

```text
mix1_keep_compatible_with_owns
mix2_keep_bought_owns
mix3_keep_bought_compatible_with
```

Their semantic meaning is:

| Internal identifier | Relations retained |
|---|---|
| `mix1_keep_compatible_with_owns` | `compatible_with` + `owns` |
| `mix2_keep_bought_owns` | `bought` + `owns` |
| `mix3_keep_bought_compatible_with` | `bought` + `compatible_with` |

### Targeted Variants

```text
targeted_pair_cw_cwm
targeted_pair_cw_instance
targeted_single_cw
targeted_triple_cw_bought_instance
targeted_triple_cw_cwm_owns
```

Their semantic meaning is:

| Internal identifier | Relations retained |
|---|---|
| `targeted_pair_cw_cwm` | `compatible_with` + `compatible_with_model` |
| `targeted_pair_cw_instance` | `compatible_with` + `instance_of` |
| `targeted_single_cw` | `compatible_with` only |
| `targeted_triple_cw_bought_instance` | `compatible_with` + `bought` + `instance_of` |
| `targeted_triple_cw_cwm_owns` | `compatible_with` + `compatible_with_model` + `owns` |

## Ablation Output Structure

Each variant contains exactly five experiment artifacts.

For example:

```text
ablation_transe/loo_no_compatible_with/
├── best_metrics_loo_no_compatible_with.json
├── best_model_loo_no_compatible_with.pt
├── context_loo_no_compatible_with.json
├── run_config_loo_no_compatible_with.json
└── run_loo_no_compatible_with.log
```

The same structure is used for all 13 variants.

The root of `ablation_transe/` additionally contains:

```text
collect_transe_ablation.py
summary.csv
summary.json
```

The summary files provide an aggregated view of the isolated TransE ablation
results.

## B2B Ablation Results

The following table reports the type-constrained Stage 1 TransE results.

| KG configuration | MRR | Hits@10 |
|---|---:|---:|
| Full five-relation graph | 0.2953 | 0.5020 |
| Without `bought` | 0.2285 | 0.4083 |
| Without `compatible_with` | 0.0151 | 0.0321 |
| Without `compatible_with_model` | 0.2962 | 0.4999 |
| Without `instance_of` | 0.2988 | 0.5082 |
| Without `owns` | 0.2931 | 0.5078 |
| `compatible_with` + `owns` | 0.2139 | 0.3778 |
| `bought` + `owns` | 0.0111 | 0.0229 |
| `bought` + `compatible_with` | 0.2938 | 0.5026 |
| `compatible_with` + `compatible_with_model` | 0.2224 | 0.3978 |
| `compatible_with` + `instance_of` | 0.2051 | 0.3629 |
| `compatible_with` only | 0.2137 | 0.3837 |
| `compatible_with` + `bought` + `instance_of` | 0.2939 | 0.5005 |
| `compatible_with` + `compatible_with_model` + `owns` | 0.2254 | 0.3999 |

The strongest degradation occurs when `compatible_with` is removed.

In particular:

```text
Full five-relation graph
MRR     = 0.2953
Hits@10 = 0.5020

Without compatible_with
MRR     = 0.0151
Hits@10 = 0.0321
```

The reduced graph retaining only:

```text
bought + owns
```

also produces a strong collapse, indicating that relation composition is more
important than simply retaining a fixed number of Knowledge Graph relations.

Several reduced configurations remain close to the complete graph.

In particular, removing `instance_of` or `owns`, retaining only `bought` and
`compatible_with`, or retaining `compatible_with`, `bought`, and `instance_of`
preserves isolated link-prediction performance close to the complete graph.

The complete Stage 1 and downstream Stage 2 ablation analysis is documented
under:

```text
ablation_study/
```

## Relationship with Other Repository Directories

The source code used to generate these results is stored mainly under:

```text
B2B_alternate_learning/
ablation_study/
SASRec/
```

The processed B2B Knowledge Graph data are stored under:

```text
data/
```

The Fashion experiments and their results are maintained separately under:

```text
fashion_generalization/
```

This separation keeps the repository-level `results/` directory focused on the
canonical B2B experiment artifacts.

## Reproducibility Notes

The repository retains:

```text
selected model checkpoints
best hyperparameters
evaluation metrics
training histories
Optuna study databases
multi-seed checkpoints
ablation checkpoints
summary files
```

where they are required to trace and reproduce the reported experiments.

Technical experiment identifiers are retained in artifact names and directory
names where necessary for reproducibility, while reader-facing result tables use
semantic descriptions of the evaluated Knowledge Graph configurations.

Historical preliminary outputs that are not part of the final experimental
pipeline are intentionally excluded from this public result structure.

Generated paths stored inside textual result files should use anonymous or
repository-independent path representations rather than user-specific
filesystem locations.
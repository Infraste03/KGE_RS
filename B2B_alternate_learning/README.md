# B2B Alternate Learning

This directory contains the implementation of the alternate-learning architecture developed for the B2B recommendation experiments.

The main idea is to jointly exploit two complementary sources of information:

1. structural knowledge encoded in a Knowledge Graph;
2. sequential purchasing behavior learned from user interaction sequences.

The architecture combines a TransE-based Knowledge Graph task and a SASRec-based sequential recommendation task through a shared item/entity embedding space.

Instead of training the two models independently and combining their scores only at inference time, the proposed approach allows both tasks to update the same representations during training.

# Main idea

The architecture consists of two tasks.

## Task A: Knowledge Graph learning

Task A models structural relationships between B2B entities using a TransE-style objective.

The Knowledge Graph contains entities such as:

```text
clients
items
machines
product models
```

and relations such as:

```text
bought
compatible_with
compatible_with_model
instance_of
owns
```

Task A receives triples:

```text
(head, relation, tail)
```

and optimizes a TransE loss using positive and negative triples.

Within the shared architecture, Task A provides auxiliary Knowledge Graph
supervision to the sequential recommendation task by updating the same entity
representations used by Task B.

## Task B: Sequential recommendation

Task B is based on SASRec.

For each user, SASRec processes the sequence of previously purchased items and produces a representation of the user's current preference state.

The model then scores candidate items through their embeddings.

Task B is optimized using a recommendation loss based on positive and negatively sampled items.

Evaluation is performed using:

```text
Recall@20
NDCG@20
```

# Shared embedding architecture

The central component of the architecture is `SharedEmbedding`.

Instead of maintaining completely independent item representations for TransE and SASRec, the joint model uses one embedding matrix shared by both tasks.

Conceptually:

```text
                    SharedEmbedding
                          |
             +------------+------------+
             |                         |
             v                         v
        Task A: TransE            Task B: SASRec
             |                         |
             v                         v
      KG triple learning       Sequential recommendation
```

Both tasks therefore directly operate on and update the same trainable entity
embedding matrix.
When Task A updates an item embedding using Knowledge Graph information, Task B sees the updated representation.

Similarly, when Task B updates an item representation through recommendation training, Task A receives the modified embedding during its next optimization step. No projection layer is used between the two tasks: TransE and SASRec operate
directly in the same shared representation space.

The two tasks therefore interact through the shared embedding space rather than through a late score-fusion mechanism.

# Unified ID space

PyKEEN, the custom Knowledge Graph code, and RecBole use different internal ID systems.

A dedicated unified ID space is therefore required.

The implementation is provided by:

```text
unified_id_space.py
```

The KG ID space is used as the primary mapping.

The original KG contains approximately:

```text
24,900 KG entities
```

including items, machines, models, and clients.

RecBole contains additional items that occur in recommendation sequences but are not represented as entities in the KG.

These items are appended to the unified space.

The resulting unified space contains:

```text
24,957 entities
21,134 items
```

including 57 items appearing in RecBole but not in the original KG entity set.

The unified mapping allows translation between:

```text
KG ID
PyKEEN ID
RecBole ID
Unified ID
```

This alignment is essential because incorrect ID mapping would silently associate pretrained embeddings with the wrong entities.

Dedicated verification scripts are available under:

```text
checks/
```

# Pretraining and warm start

The joint model is not trained entirely from scratch.

Two independently pretrained components are used.

## TransE warm start

The Knowledge Graph component is initialized from the pretrained TransE model obtained during the KGE experiments.

The main checkpoint is stored outside this source directory, for example under:

```text
results/step2_hpo/TransE/best_model.pt
```

The pretrained TransE entity embeddings initialize the corresponding rows of `SharedEmbedding`.

The TransE relation embeddings initialize the Task A relation representations.

Because the KG mapping is deterministic, the entity weights can be transferred consistently to the shared embedding space.

## SASRec warm start

Task B is initialized from the pretrained standalone SASRec model.

The corresponding checkpoint is stored separately, for example under:

```text
SASRec/pth/
```

The pretrained SASRec Transformer components are transferred to the joint recommendation model.

The shared item representation is not simply replaced by the original standalone SASRec `item_embedding`, because the objective of the architecture is to use the common KG/recommendation embedding space.

This means that the performance of the joint model immediately after warm start is not expected to be identical to standalone SASRec.

# Alternate learning

The joint architecture is trained using:

```text
alternate_loop.py
```

which implements the `AlternateTrainer`.

The trainer uses one optimizer over the complete joint model.

Therefore, updates from both tasks can affect the shared parameters.

Three scheduling strategies are implemented.

## `one_to_one`

Training alternates one Task A batch and one Task B batch:

```text
Task A batch
Task B batch
Task A batch
Task B batch
...
```

This produces frequent interaction between the Knowledge Graph and recommendation objectives.

It is the most direct form of alternate learning.

## `epoch`

A complete Task A epoch is performed first, followed by a complete Task B epoch:

```text
Task A
Task A
Task A
...
Task B
Task B
Task B
...
```

The shared representation therefore receives updates from only one objective for longer periods.

This scheduling strategy is useful for studying how the granularity of alternation affects the final recommendation performance.

## `adaptive`

The trainer dynamically chooses which task to update based on recent losses.

The task with the larger moving loss receives more updates, while a minimum number of Task B updates is preserved.


The `adaptive` strategy is implemented in the codebase but is not part of the
final experimental comparison reported in the paper, which focuses on
`one_to_one` and `epoch` scheduling.

# Step 4 training pipeline

The main training entry point is:

```text
run_step4.py
```

Its execution pipeline is:

```text
1. Load YAML configuration
2. Build the unified ID space
3. Load RecBole data
4. Build RecBole -> Unified translation
5. Build Task A and Task B DataLoaders
6. Instantiate JointAlternateModel
7. Warm-start TransE
8. Warm-start SASRec
9. Instantiate AlternateTrainer
10. Run alternate learning
11. Evaluate periodically on validation data
12. Save checkpoints and training history
13. Reload the validation-selected best model
14. Evaluate the final model on the test set
```

The main outputs include:

```text
best_model.pt
last_model.pt
checkpoint_latest.pth
checkpoint_epoch_<N>.pth
training_history.json
best_metrics.json
run.log
```

`best_model.pt` corresponds to the model selected according to validation performance.

The code may also store:

```text
best_model_on_test.pt
```

for diagnostic analysis. This checkpoint should not be used for model selection in the final experimental protocol; final reporting should be based on the validation-selected model.

# Data loading

Data-loading utilities are implemented in:

```text
data_loaders.py
```

Two PyTorch DataLoaders are constructed.

## Task A DataLoader

The Knowledge Graph TSV file is converted into integer triples:

```text
(head, relation, tail)
```

already mapped into the unified ID space.

## Task B DataLoader

RecBole sequential batches are converted into native PyTorch tensors containing:

```text
item sequence
sequence length
positive target item
```

RecBole item IDs are translated into unified IDs before being used by the joint model.

# Knowledge Graph loading

The module:

```text
load_kg.py
```

loads:

```text
kg_train.tsv
taskA_valid.tsv
taskA_test.tsv
```

and builds deterministic mappings for:

```text
entities
relations
entity types
```

Entities are grouped into:

```text
item
machine
model
client
```

The deterministic mapping is important for compatibility with the pretrained KGE checkpoints.

# Evaluation

Evaluation utilities are implemented in:

```text
eval_utils.py
```

## Task B

The main recommendation evaluation computes:

```text
Recall@20
NDCG@20
```

Candidate scores are produced for all RecBole items.

Items already contained in the user's input history are masked before ranking.

The target item is preserved so that it can correctly compete against the remaining candidate items.

The implementation also includes diagnostic checks for the masking procedure.

## Task A

A lightweight Task A evaluation utility is also provided for monitoring TransE link-prediction behavior.

More complete Step 2 reproduction checks are available under:

```text
checks/
```

# Hyperparameter optimization

Step 4 HPO is implemented in:

```text
run_hpo_step4.py
```

and uses Optuna.

The current search includes parameters such as:

```text
learning rate
Task B batch size
Task A batch size
number of Task B negatives
TransE margin
SASRec hidden dropout
SASRec attention dropout
```

Each Optuna trial:

```text
1. creates a trial-specific configuration;
2. launches `run_step4.py`;
3. collects the validation metrics;
4. stores trial metadata;
5. stores trial outputs and checkpoints;
6. records the result in CSV summaries.
```

The optimization objective is based on validation Recall@20.

HPO outputs include files such as:

```text
trials_summary.csv
best_valid.csv
best_test.csv
config_trial_<N>.yaml
trial_<N>_stdout.txt
Optuna database
```

The HPO results themselves should be stored outside the source directory, under the project `results/` or experiment-output directories.

# Post-HPO stability analysis

The script:

```text
run_hpo_stability.py
```

can rerun selected HPO configurations using several random seeds.

Its purpose is to determine whether a promising HPO configuration remains stable across independent runs.

For each seed it stores the corresponding stdout and collects:

```text
validation Recall@20
validation NDCG@20
test Recall@20
test NDCG@20
```

It then computes mean and standard deviation across seeds.

# Final B2B Results

After hyperparameter optimization, the best configuration for each scheduling strategy was retrained using five random seeds:

```text
2020
42
1024
999
2024
```

The standalone 400-dimensional SASRec model was evaluated using the same multi-seed protocol and is used as the sequential recommendation baseline.

The final results are:

| Model | Scheduling | Recall@20 | NDCG@20 |
|---|---|---:|---:|
| SASRec baseline | - | 0.3452 ± 0.0185 | 0.1840 ± 0.0074 |
| **KGSEQ** | **one_to_one** | **0.4612 ± 0.0177** | 0.2773 ± 0.0098 |
| **KGSEQ** | **epoch** | 0.4566 ± 0.0152 | **0.2783 ± 0.0101** |

Values report mean ± standard deviation across the five random seeds.

Both alternate-learning configurations substantially outperform the standalone SASRec baseline.

Compared with SASRec, the `one_to_one` configuration improves mean Recall@20 from:

```text
0.3452 -> 0.4612
```

while the `epoch` configuration reaches:

```text
Recall@20 = 0.4566
NDCG@20  = 0.2783
```

The two scheduling strategies therefore produce very similar recommendation performance.

`one_to_one` achieves the highest mean Recall@20, while `epoch` achieves the highest mean NDCG@20.


## Selected Hyperparameters

The best HPO configuration for `one_to_one` was:

| Hyperparameter | Value |
|---|---:|
| learning_rate | 0.00008 |
| batch_size_sasrec | 128 |
| batch_size_kge | 1024 |
| num_neg_b | 1 |
| margin_loss | 7.2617 |
| hidden_dropout | 0.4490 |
| attn_dropout | 0.4253 |

The best HPO configuration for `epoch` was:

| Hyperparameter | Value |
|---|---:|
| learning_rate | 0.0002 |
| batch_size_sasrec | 1024 |
| batch_size_kge | 2048 |
| num_neg_b | 1 |
| margin_loss | 1.9092 |
| hidden_dropout | 0.2688 |
| attn_dropout | 0.5122 |


## Statistical Significance

A paired one-sided Wilcoxon signed-rank test was used to compare KGSEQ against the standalone SASRec baseline across the same random seeds.

For both Recall@20 and NDCG@20:

```text
p = 0.0312
```

with:

```text
p < 0.05
```

indicating that KGSEQ consistently improves over the SASRec baseline under the adopted paired multi-seed evaluation protocol.

# KG ablation experiments

The Step 4 ablation runner is:

```text
run_step4_ablation.py
```

The ablation experiments investigate how the Knowledge Graph relations affect the final sequential recommendation model.

The selected variants include:

```text
loo_no_compatible_with
loo_no_instance_of
mix1_keep_compatible_with_owns
mix3_keep_bought_compatible_with
targeted_triple_cw_bought_instance
```

For each variant:

```text
Task A KG triples change
TransE warm-start checkpoint changes
Task B data remain unchanged
SASRec warm start remains unchanged
alternate-learning hyperparameters remain controlled
```

This makes it possible to study how the structural information provided by the KG affects downstream recommendation performance.

The current `run_step4_ablation.py` implements the corrected ablation pipeline: the original KG vocabulary is preserved when building the unified ID space, while the Task A DataLoader uses the filtered KG triples associated with the selected ablation variant.

# Statistical comparison

The repository also contains:

```text
wilcoxon_test.py
```

which performs a paired one-sided Wilcoxon signed-rank test between multi-seed results.

This is used to assess whether the recommendation performance of KGSEQ is
consistently higher than that of the standalone SASRec baseline.

Final statistical analyses should use the exact same paired seed set for the compared models.

# Additional verification utility

The script:

```text
verify_full_transe.py
```

is a diagnostic utility used to reload the full TransE checkpoint and verify its Task A performance.

More systematic verification scripts are available in:

```text
checks/
```

# Directory structure

The recommended organization is:

```text
B2B_alternate_learning/
├── README.md
├── alternate_loop.py
├── data_loaders.py
├── eval_utils.py
├── load_kg.py
├── run_hpo_stability.py
├── run_hpo_step4.py
├── run_step4.py
├── run_step4_ablation.py
├── unified_id_space.py
├── verify_full_transe.py
├── wilcoxon_test.py
│
├── configs/
│   └── README.md
│
├── models/
│   └── README.md
│
├── checks/
│   └── README.md
│
└── hpc/
    └── README.md
```

Generated experimental outputs should not normally be stored inside this source directory.

They should instead be placed under the repository-level:

```text
results/
```

# External experiment artifacts

Large or generated files are intentionally kept outside `B2B_alternate_learning/`.

Typical examples are:

```text
results/step2_hpo/
results/ablation_transe/
results/sasrec_seeds/
results/step4_hpo/
results/step4_runs/
results/step4_ablation/
```

Pretrained SASRec checkpoints may be stored under:

```text
SASRec/pth/
```

Large `.pth` and `.pt` files can be excluded from Git when appropriate, while configuration files, metric summaries, and documentation should remain available for reproducibility.

# Configuration and paths

Experiment configurations are stored under:

```text
configs/
```

Local scripts should preferably use repository-relative or file-derived paths.

HPC execution scripts are different: paths inside `hpc/*.sh` may reflect the filesystem structure of the cluster on which the original experiments were executed.

Therefore, HPC paths must be checked and adapted to the target HPC environment before running the jobs.

# Verification

Before running final experiments, the scripts under:

```text
checks/
```

can be used to verify:

```text
TransE checkpoint integrity
Step 2 metric reproducibility
KG / RecBole ID alignment
Task A scoring
Task B data coherence
padding handling
warm-start correctness
joint architecture
evaluation masking
RecBole evaluation consistency
saved Step 4 checkpoints
```

These checks are intended to detect silent mapping, initialization, or evaluation errors before expensive HPC experiments are launched.

# Experimental workflow

The complete workflow can be summarized as:

```text
Knowledge Graph
      |
      v
Pretrained TransE
      |
      +-------------------+
                          |
                          v
                    SharedEmbedding
                          ^
                          |
      +-------------------+
      |
Pretrained SASRec
      |
      v
SASRec Transformer
      |
      v
Task B
```

During Step 4:

```text
Task A loss
     |
     v
SharedEmbedding
     ^
     |
Task B loss
```

The two objectives therefore continuously influence the same entity/item representation.

This interaction between structural KG information and sequential
recommendation through a shared representation is the central idea of KGSEQ.
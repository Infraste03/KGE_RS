# B2B Alternate Learning — Verification and Diagnostic Checks

This directory contains verification, sanity-check, and diagnostic scripts used to validate the individual components of the B2B alternate-learning pipeline.

The checks cover the complete integration process between:

- the Knowledge Graph and the pretrained TransE model from Step 2;
- the pretrained SASRec model from Step 3;
- the unified entity/item ID space;
- the shared embedding matrix;
- Task A, based on TransE;
- Task B, based on SASRec;
- the joint alternate-learning architecture;
- the evaluation and masking procedures;
- the saved Step 4 checkpoints.

These scripts are primarily intended for development, reproducibility verification, and debugging. They are not a `pytest` test suite: most scripts are standalone programs that print diagnostic information and use assertions or metric comparisons to detect inconsistencies.

## Directory role

The main training code is located in:

```text
B2B_alternate_learning/
```

The model components are located in:

```text
B2B_alternate_learning/models/
```

The experiment configurations are located in:

```text
B2B_alternate_learning/configs/
```

This `checks/` directory should contain only scripts whose purpose is to verify that the data, pretrained models, ID mappings, architecture, evaluation procedure, and saved checkpoints behave as expected.

## Recommended execution location

Unless otherwise specified, run the scripts from the repository root:

```text
KG_RS/
```

For example:

```bash
python B2B_alternate_learning/checks/verify_id_alignment.py
```

Running from the repository root is recommended because some diagnostic scripts use repository-relative paths.

## Expected project paths

The checks assume the following main project structure:

```text
KG_RS/
├── B2B_alternate_learning/
│   ├── checks/
│   ├── configs/
│   ├── models/
│   ├── data_loaders.py
│   ├── eval_utils.py
│   ├── load_kg.py
│   ├── unified_id_space.py
│   └── ...
│
├── data/
│   └── processed/
│       ├── kg_train.tsv
│       ├── taskA_valid.tsv
│       └── taskA_test.tsv
│
├── dataset/
│   └── b2b_data/
│       ├── b2b_data.train.inter
│       ├── b2b_data.valid.inter
│       └── b2b_data.test.inter
│
├── SASRec/
│   └── pth/
│       └── SASRec-May-19-2026_08-28-06.pth
│
└── results/
    └── step2_hpo/
        └── TransE/
            ├── best_model.pt
            ├── best_metrics.json
            └── best_paramsTransE.json
```

Machine-specific absolute paths should not be used in these scripts.

The Step 2 TransE artifacts are expected under:

```text
results/step2_hpo/TransE/
```

The pretrained SASRec checkpoint used for the Step 3 warm start is expected under:

```text
SASRec/pth/
```

Paths used by Step 4 scripts may also be specified in:

```text
B2B_alternate_learning/configs/step4_config.yaml
```

The configuration file should use repository-relative paths rather than machine-specific paths.

## Dependencies

The checks rely on the same environment used by the main B2B experiments.

Main dependencies include:

```text
torch
pandas
numpy
pyyaml
recbole
pykeen
```

Some checks automatically use CUDA when available, while others can run entirely on CPU.

## Overview of checks

| Script | Main purpose | Category |
|---|---|---|
| `check_padding_targets.py` | Verify that RecBole targets never correspond to the padding ID | Data integrity |
| `verify_id_alignment.py` | Verify alignment between KG entities, RecBole items, and internal mappings | ID integrity |
| `verify_pretrained_weights.py` | Validate the pretrained TransE checkpoint | Pretrained model |
| `verify_step2_metrics.py` | Reproduce the Step 2 TransE metrics using native PyKEEN | Reproducibility |
| `verify_task_a_type_index.py` | Verify the Task A entity-type index | Task A sanity check |
| `debug_single_score.py` | Compare individual TransE scores between PyKEEN and the custom Task A implementation | Low-level debug |
| `test_task_a_real_data.py` | Reproduce Step 2 metrics using `SharedEmbedding + TaskATransE` | Task A integration |
| `test_task_b_real_data.py` | Check SASRec replication and Task B integration on the real dataset | Task B integration |
| `verify_task_b_history_coherence.py` | Check histories, targets, padding, and RecBole-to-unified mapping | Task B data integrity |
| `verify_architecture.py` | Verify the joint architecture after warm start | Architecture |
| `verify_evaluation_masking.py` | Inspect the masking of previously seen items | Evaluation |
| `verify_recbole_eval.py` | Compare RecBole recommendations with the custom recommendation procedure | Evaluation |
| `verify_best_checkpoint.py` | Reload and evaluate a saved Step 4 best checkpoint | Step 4 reproducibility |


## `check_padding_targets.py`

### Purpose

Checks that padding IDs are never used as recommendation targets in the RecBole train, validation, or test data loaders.

The script loads:

```text
B2B_alternate_learning/configs/step4_config.yaml
```

and creates the RecBole dataset and data loaders using the same configuration employed by the alternate-learning pipeline.

### Checks performed

For each split, the script counts:

```text
target item_id == 0
```

where RecBole ID `0` is reserved for padding.

It reports:

```text
Train targets
Valid targets
Test targets
```

and optionally checks whether empty histories are present in `user_history_dict`.

### Expected result

No real recommendation target should be padding:

```text
Train targets: 0/... are padding (0)
Valid targets: 0/... are padding (0)
Test targets: 0/... are padding (0)
```

### Run

```bash
python B2B_alternate_learning/checks/check_padding_targets.py
```

## `verify_id_alignment.py`

### Purpose

Checks the consistency between the different ID spaces used by the pipeline.

Three representations must be compatible:

1. KG entity IDs created by `load_kg.py`;
2. PyKEEN entity IDs used to train TransE;
3. RecBole item IDs used by SASRec.

Correct ID alignment is essential because the alternate-learning architecture uses hard parameter sharing through a single embedding matrix.

A mapping error would load pretrained embeddings into incorrect rows without necessarily causing a runtime error.

### Checks performed

The script:

- loads the KG mappings;
- extracts all KG entities whose label starts with `item_`;
- extracts the item tokens from the RecBole `.inter` files;
- compares the KG and RecBole item sets;
- reports items appearing only in the KG;
- reports items appearing only in RecBole;
- loads RecBole itself and inspects its internal token-to-ID mapping.

### Main paths

```text
data/processed/kg_train.tsv
data/processed/taskA_valid.tsv
data/processed/taskA_test.tsv

dataset/b2b_data/b2b_data.train.inter
dataset/b2b_data/b2b_data.valid.inter
dataset/b2b_data/b2b_data.test.inter
```

### Run

```bash
python B2B_alternate_learning/checks/verify_id_alignment.py
```

### Interpretation

An exact item-set match means that the same items are represented in both systems.

A small number of RecBole-only items can still be handled by the unified ID space, but they do not have pretrained KG embeddings and therefore require explicit handling during the construction of the shared embedding matrix.

## `verify_pretrained_weights.py`

### Purpose

Validates the pretrained TransE checkpoint before it is used for warm-starting Task A.

The expected TransE architecture is:

```text
24,900 entities
5 relations
400-dimensional embeddings
```

### Checks performed

The script verifies:

- that `best_model.pt` can be loaded;
- that the expected PyKEEN state-dict keys are present;
- the number of entities;
- the number of relations;
- the embedding dimensionality;
- the absence of `NaN` values;
- the absence of infinite values;
- entity embedding norm statistics;
- relation embedding norm statistics;
- consistency between the checkpoint entity/relation counts and `load_kg.py`.

### Expected checkpoint

```text
results/step2_hpo/TransE/best_model.pt
```

### Run

```bash
python B2B_alternate_learning/checks/verify_pretrained_weights.py
```

A successful execution indicates that the checkpoint is structurally compatible with the Task A warm-start procedure.

## `verify_step2_metrics.py`

### Purpose

Independently verifies that the saved Step 2 TransE checkpoint reproduces the metrics associated with it.

This is an important distinction from `test_task_a_real_data.py`.

`verify_step2_metrics.py` evaluates the checkpoint using native PyKEEN, while `test_task_a_real_data.py` evaluates the same weights through the custom Task A infrastructure.

Therefore:

```text
verify_step2_metrics.py
```

checks the Step 2 model itself, while:

```text
test_task_a_real_data.py
```

checks whether the custom implementation faithfully reproduces Step 2.

### Required Step 2 artifacts

```text
results/step2_hpo/TransE/best_model.pt
results/step2_hpo/TransE/best_metrics.json
results/step2_hpo/TransE/best_paramsTransE.json
```

### Evaluation

The script reconstructs a native PyKEEN `TransE` model, reloads the checkpoint, and evaluates the `compatible_with` relation using:

- naive ranking;
- type-constrained ranking;
- filtered head prediction;
- filtered tail prediction.

Metrics include:

```text
MRR
MR
Hits@1
Hits@3
Hits@10
```

The reproduced results are then compared with `best_metrics.json`.

### Run

```bash
python B2B_alternate_learning/checks/verify_step2_metrics.py
```

This check should normally be performed before debugging the custom Task A implementation.

## `verify_task_a_type_index.py`

### Purpose

Provides a lightweight check of the entity-type indexing used during type-constrained Task A evaluation.

The script specifically validates:

```python
type_index["item"]
```

### Checks performed

It verifies that the item index:

- is a PyTorch tensor;
- is one-dimensional;
- contains multiple entity IDs;
- uses `torch.long`;
- can therefore safely be used as an index tensor.

### Run

```bash
python B2B_alternate_learning/checks/verify_task_a_type_index.py
```

This is a low-cost sanity check and does not instantiate or train a model.

## `debug_single_score.py`

### Purpose

Provides a low-level diagnostic comparison between:

- the custom `TaskATransE` implementation;
- native PyKEEN `TransE`.

It is useful when the two implementations produce different ranking metrics and the discrepancy needs to be traced down to the score of an individual triple.

### Procedure

The script:

1. loads the real KG;
2. builds `SharedEmbedding`;
3. builds `TaskATransE`;
4. loads the pretrained TransE entity and relation embeddings;
5. rebuilds a native PyKEEN TransE model;
6. selects the first test triple;
7. compares embedding norms;
8. calculates the TransE distance manually;
9. compares custom and PyKEEN scores;
10. compares the top-scoring candidate entities.

It reports several distance formulations:

```text
L2 norm
squared L2 norm
L1 / Manhattan norm
```

This helps identify differences in the exact scoring function or normalization procedure.

### Required checkpoint

```text
results/step2_hpo/TransE/best_model.pt
```

### Run

```bash
python B2B_alternate_learning/checks/debug_single_score.py
```

This script is intended for debugging rather than routine experiment execution.

## `test_task_a_real_data.py`

### Purpose

Tests the custom Task A implementation on the real Step 2 data.

The objective is to verify that:

```text
SharedEmbedding + TaskATransE
```

can reproduce the metrics obtained by the pretrained PyKEEN TransE model.

### Procedure

The script:

1. loads the KG;
2. creates the custom shared embedding;
3. creates `TaskATransE`;
4. loads pretrained entity weights;
5. loads pretrained relation weights;
6. constructs type indices;
7. constructs filtering structures from train, validation, and test triples;
8. performs head prediction;
9. performs tail prediction;
10. evaluates both naive and type-constrained ranking;
11. compares the results against `best_metrics.json`.

### Step 2 reference

The expected type-constrained average MRR for the selected TransE configuration is approximately:

```text
MRR = 0.2953
```

### Required artifacts

```text
results/step2_hpo/TransE/best_model.pt
results/step2_hpo/TransE/best_metrics.json
```

### Run

```bash
python B2B_alternate_learning/checks/test_task_a_real_data.py
```

A close match with the Step 2 results provides evidence that the custom Task A infrastructure correctly reproduces the pretrained TransE model.

## `test_task_b_real_data.py`

### Purpose

Tests the SASRec component on the real recommendation data and verifies the transition from standalone SASRec to the Task B architecture.

The script includes two main tests.

### Test 1: legacy SASRec replication

`TaskBSASRecLegacy` is initialized with the original SASRec architecture and the full pretrained RecBole checkpoint.

The goal is to verify that the custom implementation can reproduce the standalone SASRec behavior.

The architecture uses:

```text
hidden size: 400
layers: 4
heads: 2
inner size: 256
hidden dropout: 0.4
attention dropout: 0.3
maximum sequence length: 50
```

### Test 2: shared-embedding Task B

The script then builds the unified item/entity ID space and creates the Task B architecture based on `SharedEmbedding`.

RecBole item IDs are translated to unified IDs before being passed to the model.

Only the appropriate pretrained SASRec components are loaded into the Task B model.

### Required SASRec checkpoint

```text
SASRec/pth/SASRec-May-19-2026_08-28-06.pth
```

### Main metric

The script computes:

```text
Recall@20
```

### Run

```bash
python B2B_alternate_learning/checks/test_task_b_real_data.py
```

This is an integration diagnostic rather than the main Step 4 evaluation script.

## `verify_task_b_history_coherence.py`

### Purpose

Checks that the recommendation evaluation data are coherent with the unified ID space and with the masking procedure used by Task B.

This is a lightweight validation and does not run training.

### Checks performed

The script verifies that:

- the validation and test RecBole loaders can be built;
- the test dataset exposes `user_history_dict`;
- validation targets are never padding;
- test targets are never padding;
- sample test users have non-empty histories;
- test targets can be represented in the unified ID space;
- the RecBole-to-unified mapping covers the evaluated targets.

### Configuration

The script automatically loads:

```text
B2B_alternate_learning/configs/step4_config.yaml
```

### Run

```bash
python B2B_alternate_learning/checks/verify_task_b_history_coherence.py
```

A successful run ends with:

```text
Task B history coherence check PASSED
```

## `verify_architecture.py`

### Purpose

Verifies the structure of `JointAlternateModel` immediately after the warm-start process and before alternate learning begins.

This check focuses on the hard-shared embedding architecture.

### Checks performed

The script verifies:

1. construction of the unified ID space;
2. successful loading of the RecBole dataset;
3. correct `SharedEmbedding` dimensions;
4. presence of an additional padding row;
5. isolation of the padding embedding;
6. loading of the pretrained TransE entity embeddings;
7. loading of the SASRec Transformer weights;
8. exclusion of the original SASRec `item_embedding` from the shared embedding matrix.

The last point is particularly important.

The architecture deliberately does not initialize the shared item representation from the standalone SASRec item embedding. The shared representation is instead initialized from the KG side for entities available in TransE and must subsequently adapt through alternate learning.

Therefore, initial recommendation performance is not expected to be identical to standalone SASRec.

### Configuration

Default:

```text
B2B_alternate_learning/configs/step4_config.yaml
```

### Run

```bash
python B2B_alternate_learning/checks/verify_architecture.py
```

or explicitly:

```bash
python B2B_alternate_learning/checks/verify_architecture.py \
    --config B2B_alternate_learning/configs/step4_config.yaml
```

## `verify_evaluation_masking.py`

### Purpose

Inspects the recommendation masking logic used during Task B evaluation.

In full-sort recommendation, items already present in a user's history must be excluded from the candidate ranking.

### Procedure

The script:

1. loads the unified ID space;
2. loads the validation data;
3. initializes the warm-started joint model;
4. selects one user with a sufficiently long sequence;
5. computes raw recommendation scores;
6. identifies items already seen by the user;
7. prints selected scores before masking;
8. masks previously seen items with `-inf`;
9. prints scores after masking;
10. verifies that the target remains available;
11. computes the final target rank.

### Configuration

Default:

```text
B2B_alternate_learning/configs/step4_config.yaml
```

### Run

```bash
python B2B_alternate_learning/checks/verify_evaluation_masking.py
```

This script is primarily useful when investigating suspicious Recall or NDCG values.

## `verify_recbole_eval.py`

### Purpose

Provides a direct recommendation-level comparison between:

- the original RecBole SASRec model;
- the recommendation procedure used by the joint architecture.

Rather than comparing only aggregate metrics, the script compares the actual Top-20 recommendation lists for a selected user.

### Procedure

For the selected user, it:

1. loads the original SASRec checkpoint;
2. obtains the Top-20 items using RecBole;
3. constructs the unified ID space;
4. translates RecBole item IDs to unified IDs;
5. obtains the Top-20 items from the custom recommendation path;
6. applies history masking;
7. compares the two recommendation lists.

The default diagnostic user is:

```text
USER_ID_TO_TEST = 111
```

### Required SASRec checkpoint

```text
SASRec/pth/SASRec-May-19-2026_08-28-06.pth
```

### Configuration

```text
B2B_alternate_learning/configs/step4_config.yaml
```

### Run

```bash
python B2B_alternate_learning/checks/verify_recbole_eval.py
```

This script is intended as a detailed diagnostic when the custom evaluation procedure and RecBole appear to disagree.

## `verify_best_checkpoint.py`

### Purpose

Verifies that a Step 4 checkpoint saved at the best validation epoch can be reloaded and reproduces the expected metrics.

This should be used before launching final multi-seed runs.

### Procedure

The script:

1. loads a Step 4 configuration;
2. overrides the RecBole seed using the CLI argument;
3. reconstructs the unified ID space;
4. reconstructs the RecBole validation and test loaders;
5. rebuilds `JointAlternateModel` using the same architecture;
6. reloads the saved checkpoint;
7. evaluates validation Recall@20 and NDCG@20;
8. evaluates test Recall@20 and NDCG@20;
9. optionally compares validation Recall@20 with an expected value.

### Example

```bash
python B2B_alternate_learning/checks/verify_best_checkpoint.py \
    --config B2B_alternate_learning/configs/step4_seeds_one_to_one.yaml \
    --checkpoint <path-to-step4-best-model.pt> \
    --seed 2020 \
    --expected-valid-r20 0.5000
```

The checkpoint path is intentionally passed as a command-line argument because different HPO trials and seed runs generate different Step 4 checkpoints.

### Reproducibility criterion

When `--expected-valid-r20` is provided, the script reports a successful reproducibility check when:

```text
|actual Recall@20 - expected Recall@20| < 0.001
```



## Recommended verification order

For a fresh environment or a clean reproduction of the B2B alternate-learning pipeline, the recommended sequence is:

```text
1. verify_pretrained_weights.py
2. verify_step2_metrics.py
3. verify_id_alignment.py
4. verify_task_a_type_index.py
5. check_padding_targets.py
6. verify_task_b_history_coherence.py
7. debug_single_score.py             [only if Task A scores need debugging]
8. test_task_a_real_data.py
9. test_task_b_real_data.py
10. verify_architecture.py
11. verify_evaluation_masking.py
12. verify_recbole_eval.py            [only if recommendation evaluation needs debugging]
13. verify_best_checkpoint.py
```



before warm-starting Step 4 from an ablation-specific TransE checkpoint.

## Relationship between the main checks

Several scripts intentionally overlap, but they test different layers of the pipeline.

### Step 2 verification

```text
verify_pretrained_weights.py
        |
        v
verify_step2_metrics.py
```

The first validates the checkpoint structure.

The second validates its native PyKEEN performance.

### Task A integration

```text
Step 2 TransE checkpoint
        |
        v
SharedEmbedding
        |
        v
TaskATransE
        |
        v
test_task_a_real_data.py
```

If native PyKEEN evaluation is correct but `test_task_a_real_data.py` differs substantially, the issue is likely in the custom Task A implementation, mapping, scoring, normalization, or filtering.

`debug_single_score.py` can then be used to investigate individual scores.

### Task B integration

```text
Pretrained SASRec
        |
        v
TaskBSASRec
        |
        v
Unified ID mapping
        |
        v
Task B evaluation
```

The relevant checks are:

```text
test_task_b_real_data.py
verify_task_b_history_coherence.py
verify_evaluation_masking.py
verify_recbole_eval.py
```

### Full architecture

Once Task A and Task B have independently been checked:

```text
verify_architecture.py
```

validates their integration in `JointAlternateModel`.

Finally:

```text
verify_best_checkpoint.py
```

checks that the resulting Step 4 model can be saved, reloaded, and evaluated reproducibly.

## Notes on reproducibility

The checks are intended to catch silent inconsistencies that can strongly affect the final experimental results, including:

- mismatched PyKEEN and RecBole IDs;
- incorrect item-to-unified-ID translation;
- incorrect padding handling;
- incorrect pretrained embedding loading;
- accidental loading of SASRec item embeddings into the shared embedding;
- inconsistent TransE scoring;
- incorrect type constraints;
- incorrect filtered ranking;
- incorrect masking of user history;
- checkpoint/configuration mismatches;
- non-reproducible Step 4 checkpoint evaluation.

A successful run of an individual check confirms only the specific property tested by that script. The recommended sequence above should therefore be used when validating the complete pipeline.

## Scope

These scripts are diagnostic utilities and should not be used as the primary training entry points.

The actual training, hyperparameter optimization, multi-seed execution, and final evaluation procedures are implemented in the parent `B2B_alternate_learning/` directory and its HPC scripts.

The `checks/` directory exists to make the alternate-learning pipeline easier to inspect, debug, and reproduce.
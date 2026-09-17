# Fashion-Random Alternate Learning - Verification Checks

This directory contains the verification and diagnostic scripts used to validate
the Fashion-Random alternate-learning pipeline before running expensive training
experiments.

The checks cover the complete integration path between:

- the Fashion Knowledge Graph;
- PyKEEN / TransE;
- RecBole / SASRec;
- the unified ID space;
- the shared embedding layer;
- Task A;
- Task B;
- the final `JointAlternateModelFashion`.

The main goal of these scripts is to detect silent inconsistencies before alternate learning starts.

In particular, the checks verify that:

1. Knowledge Graph entities are mapped consistently;
2. RecBole item IDs are deterministic and aligned with KG items;
3. pretrained TransE weights are valid and reproducible;
4. pretrained SASRec weights are valid and reproducible;
5. Task A reproduces the expected standalone TransE behavior;
6. Task B reproduces the expected standalone SASRec behavior;
7. the shared embedding receives the correct TransE initialization;
8. the SASRec Transformer is correctly warm-started;
9. evaluation protocols are consistent with the original HPO experiments;
10. the integrated joint model is ready for alternate learning.


# Directory Contents

```text
checks/
├── README.md
├── check_recbole_mapping_fashion.py
├── inspect_sasrec_checkpoint_fashion.py
├── test_task_a_real_data_fashion.py
├── test_task_b_real_data_fashion.py
├── verify_id_alignment_fashion.py
├── verify_pretrained_weights_fashion.py
├── verify_sasrec_metrics_fashion.py
├── verify_shared_embedding_warmstart_fashion.py
├── verify_step2_metrics_fashion.py
├── verify_task_a_type_index_fashion.py
├── verify_task_b_history_coherence_fashion.py
├── verify_transe_metrics_fashion.py
└── verify_warmstart_fashion.py
```


# Reference Fashion-Random Setup

The verification scripts assume the Fashion-Random alternate-learning
configuration used in the experiments.

## Knowledge Graph

The Fashion KG contains three entity types:

```text
item
category
brand
```

and three relation types.

The main relation evaluated by Task A is:

```text
compatible_with
```

which is an:

```text
item -> item
```

relation.

The Fashion KG contains:

```text
182,983 KG entities
3 relations
```

The joint model adds one dedicated padding row:

```text
Unified padding ID = 182,983
SharedEmbedding rows = 182,984
```

The embedding dimensionality is:

```text
64
```


## SASRec

The standalone SASRec model used as warm start corresponds to the best Fashion HPO configuration.

Main architecture:

```text
hidden_size = 64
n_layers = 4
n_heads = 4
inner_size = 256
MAX_ITEM_LIST_LENGTH = 50
hidden_dropout_prob = 0.5
attn_dropout_prob = 0.3
loss_type = CE
```

The RecBole evaluation protocol is:

```text
split:
    LS: valid_and_test

order:
    TO

group_by:
    user

mode:
    valid: full
    test: full
```

The main SASRec checkpoint is expected under:

```text
fashion_generalization/results/sasrec_hpo/trial_017/best_valid.pth
```

Some verification scripts additionally support:

```text
fashion_generalization/results/sasrec_hpo/trial_017/best_valid.pth
```

when the HPO results are organized under a dedicated `sasrec_hpo` directory.


## TransE

The pretrained Fashion TransE checkpoint is expected under:

```text
fashion_generalization/results/hpo_transe/best_model.pt
```

The embedding dimensionality is:

```text
64
```

The pretrained entity embeddings therefore have expected shape:

```text
(182983, 64)
```

and the relation embeddings have expected shape:

```text
(3, 64)
```


# Reference Standalone Metrics

The verification scripts use the following values as reference points.

## Task A - TransE

For `compatible_with` on the Fashion test set:

| Metric | Expected |
|---|---:|
| MRR | 0.1057 |
| Hits@1 | 0.0691 |
| Hits@3 | 0.1183 |
| Hits@10 | 0.1755 |

Because `compatible_with` is `item -> item`, both head and tail candidates belong to the same entity type.

Therefore, for this relation, the Fashion type-constrained evaluation uses:

```text
head candidates = items
tail candidates = items
```

This is different from the B2B setting, where `compatible_with` connects items and machines.


## Task B - SASRec

The reference standalone SASRec values are:

| Split / Metric | Expected |
|---|---:|
| Validation NDCG@20 | 0.1086 |
| Test Recall@20 | 0.0964 |
| Test NDCG@20 | 0.0889 |

Some older diagnostic code contains a historical Recall@20 value around `0.0951`.

The Task B verification script avoids relying only on this hard-coded value by evaluating the checkpoint again with the official RecBole `Trainer` and using the reproduced result as the runtime reference.


# Recommended Verification Order

The scripts are most useful when executed from basic structural checks to complete integration checks.

A recommended order is:

```text
1.  inspect_sasrec_checkpoint_fashion.py
2.  verify_pretrained_weights_fashion.py

3.  verify_id_alignment_fashion.py
4.  check_recbole_mapping_fashion.py
5.  verify_task_a_type_index_fashion.py
6.  verify_task_b_history_coherence_fashion.py

7.  verify_transe_metrics_fashion.py
8.  verify_step2_metrics_fashion.py
9.  verify_sasrec_metrics_fashion.py

10. verify_shared_embedding_warmstart_fashion.py

11. test_task_a_real_data_fashion.py
12. test_task_b_real_data_fashion.py

13. verify_warmstart_fashion.py
```

The final script, `verify_warmstart_fashion.py`, should only be considered meaningful after the lower-level checks pass.


# 1. `inspect_sasrec_checkpoint_fashion.py`

This is a lightweight checkpoint-inspection utility.

It loads:

```text
results/sasrec_hpo/trial_017/best_valid.pth
```

and reports the tensors contained in the SASRec state dictionary.

It extracts information such as:

```text
number of items
hidden size
maximum sequence length
number of Transformer layers
inner size
```

Some architectural information cannot be inferred from tensor shapes alone.

For example:

```text
n_heads
```

must be recovered from the HPO configuration or HPO CSV because the query, key, and value matrices have shape:

```text
(hidden_size, hidden_size)
```

independently of the number of attention heads.

This script is useful as the first checkpoint sanity check.


# 2. `verify_pretrained_weights_fashion.py`

This script verifies both pretrained components used by the joint model.

## TransE checks

It verifies:

```text
checkpoint exists
expected state_dict keys
entity embedding shape
relation embedding shape
embedding dimension
number of entities
number of relations
absence of NaN
absence of Inf
entity norm statistics
consistency with load_kg_fashion
```

Expected TransE dimensions:

```text
entities:   182,983 x 64
relations:        3 x 64
```


## SASRec checks

It verifies:

```text
checkpoint exists
item embedding exists
hidden_size = 64
RecBole item count matches KG item count
absence of NaN
absence of Inf
```

The Fashion setting is expected to have complete item overlap between the recommendation dataset and the KG.


# 3. `verify_id_alignment_fashion.py`

This script verifies consistency between the three ID systems involved in the Fashion architecture:

```text
load_kg_fashion
PyKEEN
RecBole
```

It compares:

```text
KG item entities
Fashion .inter item tokens
RecBole internal item mapping
```

For Fashion, the expected condition is:

```text
KG items == RecBole items
```

Unlike the B2B case, Fashion does not require additional recommendation-only items outside the Knowledge Graph.

The expected result is therefore perfect item-set alignment.


# 4. `check_recbole_mapping_fashion.py`

This script verifies that RecBole produces a deterministic mapping:

```text
item token -> RecBole internal ID
```

The same RecBole dataset is constructed twice with the same configuration.

The script then verifies:

```text
same number of entries
same tokens
same internal IDs
PAD ID = 0
```

This is important because the unified ID translation depends on the RecBole internal mapping.

A silent change in the RecBole configuration could otherwise associate items with incorrect unified IDs.

The RecBole configuration used here must therefore remain consistent with the configurations used by:

```text
unified_id_space_fashion.py
data_loaders_fashion.py
```


# 5. `verify_task_a_type_index_fashion.py`

This script verifies the entity-type index used for type-constrained Task A evaluation.

The expected Fashion entity types are:

```text
item
category
brand
```

The script verifies that:

```text
each type is non-empty
each type is represented by a 1D LongTensor
the three sets are disjoint
the three sets cover all KG entities
```

It additionally verifies the semantics of `compatible_with`.

For Fashion:

```text
compatible_with:
item -> item
```

Therefore:

```text
head candidates = item
tail candidates = item
```

This differs from the B2B Knowledge Graph, where compatibility connects different entity types.


# 6. `verify_task_b_history_coherence_fashion.py`

This script validates the RecBole validation and test dataloaders used by Task B.

It deliberately does not rely on internal fields such as:

```text
history_index
user_history_dict
```

because their availability depends on the RecBole version and dataloader implementation.

Instead, it verifies that the evaluation data are constructed using exactly the same protocol used during SASRec HPO.

Checks include:

```text
valid/test dataloaders are non-empty
target item is never padding
item_id_list exists
item_length exists
sequence lengths are coherent
target items map to KG items
sequence items map to KG items
RecBole items are fully represented in the KG
```

It also verifies explicitly that the evaluation configuration is:

```text
split = valid_and_test
order = TO
group_by = user
valid mode = full
test mode = full
```

This check is important because changing the RecBole evaluation configuration would make the reproduced SASRec metrics incomparable with the original HPO results.


# 7. `verify_transe_metrics_fashion.py`

This script reloads the original TransE checkpoint directly into a PyKEEN `TransE` model.

It reconstructs the original train, validation, and test `TriplesFactory` objects and evaluates the `compatible_with` relation.

The main target is:

```text
MRR ≈ 0.1057
```

The script computes:

```text
MRR
Hits@1
Hits@3
Hits@10
```

for:

```text
tail prediction
head prediction
average head + tail
```

The evaluation is filtered using known positive triples from train, validation, and test.

The script checks that the stored `best_model.pt` corresponds to the performance reported during TransE HPO.


# 8. `verify_step2_metrics_fashion.py`

This is a more complete Step 2 reproduction check.

Compared with the lighter TransE verification, it additionally supports:

```text
best_metrics.json
best_metricsTransE.json
best_params.json
best_paramsTransE.json
```

when available.

It reconstructs the PyKEEN model from the checkpoint and evaluates:

```text
naive
type_constrained
```

modes.

It then compares reproduced metrics with the stored HPO metrics.

When no metrics JSON is available, the script uses:

```text
EXPECTED_TYPE_CONSTRAINED_MRR_FROM_LOG = 0.1057
```

as a minimum reference check.

This script is intended to verify that the final Step 2 checkpoint remains reproducible independently of the original HPO execution.


# 9. `verify_sasrec_metrics_fashion.py`

This script reproduces the standalone SASRec HPO result using the original RecBole model and evaluation pipeline.

It reads the parameters of:

```text
trial 17
```

from the HPO CSV and reconstructs the SASRec configuration.

The expected values are:

```text
Validation NDCG@20 = 0.1086
Test Recall@20     = 0.0964
Test NDCG@20       = 0.0889
```

The script loads the SASRec checkpoint and evaluates it using:

```text
Trainer._valid_epoch
```

for both validation and test data.

This is the main verification that:

```text
checkpoint
HPO configuration
dataset
RecBole evaluation protocol
```

still reproduce the original standalone SASRec result.


# 10. `verify_shared_embedding_warmstart_fashion.py`

This script focuses exclusively on the transfer of pretrained TransE entity embeddings into `SharedEmbedding`.

It verifies that:

```text
182,983 KG entity rows are loaded
embedding dimension is 64
padding row remains zero
no NaN values are present
no Inf values are present
entity norms match the TransE checkpoint
```

The script also performs exact comparisons for sample entities belonging to:

```text
brand
category
item
```

For each sampled entity it verifies that the vector stored in `SharedEmbedding` is exactly equal to the corresponding PyKEEN TransE vector.

This isolates the shared-embedding initialization step from the rest of the
joint architecture.


# 11. `test_task_a_real_data_fashion.py`

This script verifies the custom `TaskATransE` implementation using the real Fashion data and real pretrained TransE weights.

It tests whether:

```text
SharedEmbedding
+
TaskATransE
```

can reproduce the expected standalone PyKEEN Task A metrics.

Reference values:

```text
MRR     = 0.1057
Hits@1  = 0.0691
Hits@3  = 0.1183
Hits@10 = 0.1755
```

This check verifies that:

```text
PyKEEN -> custom Task A
```

weight transfer and scoring are correct.

If this check fails, alternate learning should not be started because Task A would not represent the intended pretrained KGE model.


# 12. `test_task_b_real_data_fashion.py`

This script verifies the custom Task B implementation against RecBole.

It contains two complementary tests.


## Test 1 - Legacy SASRec

`TaskBSASRecLegacy` keeps its own item embedding and loads all weights from the RecBole checkpoint.

The purpose is to verify that the custom SASRec Transformer implementation can reproduce the original RecBole behavior.

The script also evaluates the checkpoint with the official RecBole Trainer and uses that result as the runtime ground truth.

It verifies exact transfer of:

```text
position embedding
Transformer parameters
item embedding
```


## Test 2 - Shared-Embedding SASRec

`TaskBSASRec` uses:

```text
SharedEmbedding
```

instead of the original SASRec item embedding.

Only the Transformer-related SASRec parameters are warm-started.

At this stage the shared embedding is intentionally not the original standalone SASRec item embedding.

Therefore, the shared-embedding model is not expected to reproduce standalone SASRec metrics before alternate learning.

The purpose of this second test is to verify that:

```text
the Transformer was loaded correctly
the unified ID translation works
the hybrid scoring pipeline runs correctly
performance remains above a random baseline
```

This distinction is important when interpreting the warm-start behavior of the final model.


# 13. `verify_warmstart_fashion.py`

This is the final integration check.

It instantiates:

```text
JointAlternateModelFashion
```

and applies both pretrained components:

```text
TransE -> SharedEmbedding + Task A relations
SASRec -> Task B Transformer components
```

It verifies the complete initialization pipeline before alternate learning.


## Weight integrity checks

The script verifies that:

```text
SharedEmbedding entity weights match PyKEEN
padding row is zero
SASRec position embedding matches the checkpoint
Transformer query weights match the checkpoint
SASRec item_embedding is NOT copied into SharedEmbedding
```


## Task A check

Task A is expected to reproduce approximately:

```text
MRR = 0.1057
```

with a tolerance of:

```text
0.005
```


## Task B check

Task B does not use the original standalone SASRec item embedding.

The shared item representations come from the TransE warm start.

Therefore, the integrated Task B model is not expected to exactly reproduce:

```text
Recall@20 = 0.0964
```

before alternate learning.

Instead, the script checks that:

```text
the Transformer has been correctly loaded
the evaluation pipeline works
Recall@20 remains meaningfully above random
```

This is the expected behavior of the shared-embedding architecture before
alternate learning.

# Why Task B Does Not Exactly Match Standalone SASRec After the Shared Warm Start

This point is fundamental for interpreting the checks.

Standalone SASRec computes recommendation scores using:

```text
SASRec user representation
        x
SASRec item embeddings
```

The joint Fashion model instead starts from:

```text
SASRec Transformer
        x
TransE-initialized SharedEmbedding
```

Therefore, immediately after warm start, the user representation and candidate item representations do not correspond to the exact same latent geometry used by the original standalone SASRec checkpoint.

The purpose of alternate learning is precisely to adapt this shared space using both:

```text
Task A structural supervision
Task B recommendation supervision
```

Consequently:

```text
Task A should reproduce pretrained TransE immediately.
```

but:

```text
Task B is not required to reproduce standalone SASRec immediately
when SharedEmbedding contains TransE entity vectors.
```

The exact standalone SASRec reproduction is instead verified separately by:

```text
verify_sasrec_metrics_fashion.py
test_task_b_real_data_fashion.py - Legacy mode
```


# Path Convention

The verification scripts do not rely on machine-specific absolute paths.

They derive the Fashion project root from their own location:

```text
fashion_generalization/
```

and construct paths relative to it.

Expected data locations include:

```text
fashion_generalization/
├── data/
│   ├── processed/
│   │   ├── kg_train.tsv
│   │   ├── taskA_valid.tsv
│   │   └── taskA_test.tsv
│   │
│   └── recbole/
│       └── fashion/
│           └── fashion.inter
│
└── results/
    ├── hpo_transe/
    │   └── best_model.pt
    │
    └── trial_017/
        └── best_valid.pth
```

Some scripts also support an alternative SASRec result organization:

```text
results/sasrec_hpo/
```

No user-specific local path should be required.


# Running the Checks

From the repository root:

```bash
python fashion_generalization/alternate_learning/checks/check_recbole_mapping_fashion.py
```

The same convention can be used for every script.

For example:

```bash
python fashion_generalization/alternate_learning/checks/verify_id_alignment_fashion.py

python fashion_generalization/alternate_learning/checks/verify_pretrained_weights_fashion.py

python fashion_generalization/alternate_learning/checks/verify_transe_metrics_fashion.py

python fashion_generalization/alternate_learning/checks/verify_sasrec_metrics_fashion.py

python fashion_generalization/alternate_learning/checks/verify_shared_embedding_warmstart_fashion.py

python fashion_generalization/alternate_learning/checks/test_task_a_real_data_fashion.py

python fashion_generalization/alternate_learning/checks/test_task_b_real_data_fashion.py

python fashion_generalization/alternate_learning/checks/verify_warmstart_fashion.py
```


# Dependencies

The checks rely primarily on:

```text
Python
PyTorch
pandas
PyKEEN
RecBole
```

They also import modules from:

```text
fashion_generalization/alternate_learning/
```

and:

```text
fashion_generalization/alternate_learning/models/
```


# Interpretation of Failures

A failed verification should generally be investigated before running new alternate-learning experiments.

Typical failure classes include:

| Failure | Possible cause |
|---|---|
| KG entity count mismatch | different KG version |
| RecBole item mismatch | different `.inter` file or configuration |
| Mapping mismatch | RecBole configuration changed |
| TransE metric mismatch | wrong checkpoint, mapping, scoring, or evaluation protocol |
| SASRec metric mismatch | wrong checkpoint, HPO parameters, dataset, or RecBole configuration |
| SharedEmbedding mismatch | incorrect PyKEEN-to-unified mapping |
| Padding row not zero | incorrect shared embedding initialization |
| Transformer weight mismatch | incorrect SASRec warm start |
| Task A integration mismatch | incorrect Task A scoring or relation transfer |
| Task B near random | incorrect ID translation, Transformer loading, candidate construction, or masking |


# Final Verification Goal

Before running alternate learning, the complete verification chain should establish that:

```text
Fashion KG
    |
    v
TransE checkpoint
    |
    v
Task A / SharedEmbedding
    |
    | verified
    v
JointAlternateModelFashion
    ^
    |
    | verified
    |
SASRec Transformer
    ^
    |
SASRec checkpoint
```

and independently:

```text
RecBole item IDs
      |
      v
Unified ID mapping
      |
      v
Shared item space
```

The expected final state is:

```text
TransE checkpoint validated
SASRec checkpoint validated
KG mappings validated
RecBole mappings validated
Task A semantics validated
Task B dataloaders validated
SharedEmbedding warm start validated
SASRec Transformer warm start validated
Joint model integration validated
```

Only after these checks pass should the Fashion alternate-learning training pipeline be considered ready for final experiments.
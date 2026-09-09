# Fashion Alternate Learning - Models

This directory contains the model components used by the Fashion alternate-learning architecture.

The implementation combines:

- a TransE-based Knowledge Graph task;
- a SASRec-based sequential recommendation task;
- a hard-shared entity embedding matrix;
- task-specific loss functions;
- negative samplers;
- a joint model that orchestrates the two tasks.

The central idea is that Task A and Task B operate on the same `SharedEmbedding` instance.

```text
Task A: TransE
          \
           \
            SharedEmbedding
           /
          /
Task B: SASRec
```

Updates produced by either task can therefore modify the same item representations.


# Directory Contents

```text
models/
├── README.md
├── joint_model_fashion.py
├── losses_fashion.py
├── samplers_fashion.py
├── shared_embedding_fashion.py
├── task_a_transe_fashion.py
└── task_b_sasrec_fashion.py
```


# Architecture Overview

The Fashion architecture uses a 64-dimensional shared latent space.

The main dimensions are:

```text
KG entities          = 182,983
Unified padding row  = 1
SharedEmbedding rows = 182,984

Embedding dimension  = 64
KG relations         = 3

SASRec layers        = 4
SASRec heads         = 4
SASRec inner size    = 256
Max sequence length  = 50
```

The unified padding ID is:

```text
182,983
```

This differs from standard RecBole SASRec, where padding has internal ID `0`.


# Unified Entity Space

The Fashion Knowledge Graph contains three entity types:

```text
brand
category
item
```

The deterministic KG entity ordering is:

```text
brand_*    -> IDs 0..17510
category_* -> IDs 17511..17513
item_*     -> IDs 17514..182982
```

Therefore:

```text
Brands      = 17,511
Categories  = 3
Items       = 165,469
KG entities = 182,983
```

A dedicated additional row is reserved for SASRec padding:

```text
padding_idx = 182983
```

The final shared embedding matrix therefore has:

```text
182,984 x 64
```

parameters before considering the task-specific components.


# `shared_embedding_fashion.py`

`SharedEmbedding` is the core component of the hard-sharing architecture.

It owns the entity embedding matrix used by both:

```text
Task A -> TransE
Task B -> SASRec
```

The same physical object is passed to both tasks.

This is important because the architecture is designed so that updates from either task affect the same item vectors.


## Padding

RecBole normally uses:

```text
PAD = 0
```

However, Fashion unified ID `0` is already assigned to a real KG entity.

Therefore, the joint model uses:

```text
unified padding = 182983
```

The padding row:

- is initialized to zero;
- remains zero after warm start;
- does not receive gradients.


## TransE warm start

The main initialization method is:

```python
load_from_pykeen_transe(...)
```

It copies pretrained PyKEEN TransE entity embeddings into `SharedEmbedding`.

The copy is performed according to entity labels rather than assuming that PyKEEN and the custom KG loader use identical numerical IDs.

Expected source tensor:

```text
shape = (182983, 64)
```


## SASRec item warm start

The optional method:

```python
load_from_sasrec_recbole(...)
```

can overwrite item rows using standalone SASRec item embeddings.

This method is not part of the default architecture.

The standard joint model initializes the shared entity space from TransE and only imports the SASRec Transformer components.

The SASRec-item initialization method is retained for ablation experiments.


# `task_a_transe_fashion.py`

`TaskATransE` implements Task A of alternate learning.

It operates on the shared entity embeddings and owns only the relation embeddings.

```text
Shared entity embeddings -> SharedEmbedding
Relation embeddings      -> TaskATransE
```

With three relations and 64-dimensional embeddings, the task-specific relation matrix has shape:

```text
3 x 64
```


## TransE scoring

During training, the task computes the TransE energy:

```text
||h + r - t||_p
```

with:

```text
p_norm = 1
normalize_entities = True
```

Lower energy indicates a more plausible triple.


## Entity normalization

Entity representations are L2-normalized before scoring.

This behavior is required to reproduce the pretrained Fashion PyKEEN TransE model.


## Link prediction

The model provides:

```python
score_all_tails(...)
score_all_heads(...)
```

for link-prediction evaluation.

These methods return:

```text
similarity = -energy
```

so that higher values correspond to more plausible candidates.


## Relation warm start

The method:

```python
load_relation_weights_from_pykeen(...)
```

loads pretrained TransE relation embeddings.

As with the entity embeddings, relations are matched using relation labels rather than relying exclusively on numerical IDs.


# `task_b_sasrec_fashion.py`

`TaskBSASRec` implements the sequential recommendation component.

It reproduces the main RecBole SASRec Transformer architecture while replacing the standalone SASRec item embedding matrix with `SharedEmbedding`.


## Architecture

The selected Fashion configuration uses:

```text
hidden_size    = 64
n_layers       = 4
n_heads        = 4
inner_size     = 256
hidden_dropout = 0.5
attn_dropout   = 0.3
max_seq_length = 50
```


## Shared item embeddings

Unlike standalone SASRec:

```text
item_embedding
```

is not owned by Task B.

Instead:

```text
item embedding = SharedEmbedding[item_unified_id]
```

The following components remain private to Task B:

```text
position embedding
Transformer encoder
LayerNorm
dropout
```


## Attention mask

The Fashion implementation must use:

```python
item_seq != unified_padding_idx
```

rather than:

```python
item_seq > 0
```

because unified ID `0` is a valid KG entity.

The Fashion padding ID is:

```text
182983
```


## Recommendation scoring

After producing a user representation, Task B scores candidate items using a dot product:

```text
score(user, item) = user_repr · item_embedding
```

Candidate items are passed explicitly as unified IDs.

For the standard Fashion dataset, the candidate item range is:

```text
17514..182982
```

for a total of:

```text
165,469 items
```


## SASRec warm start

The method:

```python
load_self_attention_from_recbole(...)
```

loads the pretrained standalone SASRec weights for:

```text
position_embedding
Transformer layers
LayerNorm
```

but intentionally skips:

```text
item_embedding.weight
```

because item representations are stored in `SharedEmbedding` and initialized from TransE.


# `TaskBSASRecLegacy`

`task_b_sasrec_fashion.py` also contains:

```python
TaskBSASRecLegacy
```

This class reproduces the standalone RecBole architecture more directly.

Unlike the hybrid Task B, it owns its own:

```text
item_embedding
```

with:

```text
padding_idx = 0
```

It is intended only for verification and sanity checks.

Its purpose is to verify that the custom SASRec implementation can reproduce the behavior of the original standalone RecBole checkpoint before replacing the item embedding matrix with the shared TransE-initialized representation.

It is not used during alternate-learning training.


# `joint_model_fashion.py`

`JointAlternateModelFashion` combines the shared representation and the two tasks.

The structure is:

```text
JointAlternateModelFashion
│
├── SharedEmbedding
│
├── TaskATransE
│   └── relation embeddings
│
└── TaskBSASRec
    ├── position embedding
    ├── Transformer encoder
    ├── LayerNorm
    └── dropout
```


## Hard sharing

The model passes the exact same `SharedEmbedding` instance to:

```text
TaskATransE
TaskBSASRec
```

rather than creating independent copies.

This enables gradients from both tasks to update the shared item representations.


## Task A interface

The joint model exposes:

```python
forward_kge(...)
score_all_tails_kge(...)
score_all_heads_kge(...)
```


## Task B interface

For recommendation, it exposes:

```python
forward_sasrec(...)
score_items_sasrec(...)
```


## Warm start interface

The joint model provides:

```python
load_pretrained_kge(...)
```

which initializes:

```text
SharedEmbedding <- PyKEEN entity embeddings
Task A relations <- PyKEEN relation embeddings
```

and:

```python
load_pretrained_sasrec(...)
```

which initializes:

```text
Task B Transformer components <- RecBole SASRec
```

while leaving the shared item embeddings initialized from TransE.


# Warm-Start Design

The standard Fashion initialization can be summarized as:

```text
PyKEEN TransE checkpoint
        |
        +----------------------------+
        |                            |
        v                            v
SharedEmbedding              Task A relations
(entity embeddings)          (relation embeddings)
        |
        |
        +----------------------+
                               |
                               v
                     Joint shared space


RecBole SASRec checkpoint
        |
        v
position embedding
Transformer encoder
LayerNorm
        |
        v
Task B
```

Importantly:

```text
RecBole item_embedding.weight
```

is not loaded into the shared matrix in the standard experiment.


# `losses_fashion.py`

This module contains the losses used by the two tasks.


## Task A - TransE

Task A uses:

```python
TransELoss
```

which implements margin ranking:

```text
max(
    distance(pos)
    - distance(neg)
    + margin,
    0
)
```

The current default margin in the implementation is:

```text
8.684
```

The actual experiment configuration may override this value.


# Task B - BPR

The BPR variant uses:

```python
SASRecBPRLoss
```

with:

```text
-loss sigmoid(score_positive - score_negative)
```

conceptually encouraging:

```text
score_positive > score_negative
```

This variant requires negative item sampling.


# Task B - Cross Entropy

The CE variant uses:

```python
SASRecCELoss
```

and performs full-softmax classification over all candidate items.

The input scores have shape:

```text
(batch_size, num_candidate_items)
```

and the targets are:

```text
positions inside the candidate array
```

not unified entity IDs.

This distinction is important.

For example, when candidates are:

```text
17514..182982
```

a target unified ID must first be converted to its position in this candidate vector.


# `samplers_fashion.py`

This module contains the negative samplers used during training.


## `NegativeSamplerKGE`

Used for Task A.

It randomly corrupts either:

```text
head
```

or:

```text
tail
```

with approximately equal probability.

Sampling is performed over:

```text
0..182982
```

which contains only real KG entities.

The unified padding row:

```text
182983
```

is therefore excluded automatically.


## `NegativeSamplerRec`

Used for BPR-based Task B training.

It samples negative recommendation items only from:

```text
17514..182982
```

therefore excluding:

```text
brands
categories
padding
```

The sampler also ensures that a sampled negative item is not identical to the positive target.

It supports:

```text
1 negative per positive
```

or multiple negatives:

```text
(B, num_neg)
```


# BPR vs CE Task B Training

The model directory supports both Task B objectives.

For BPR:

```text
TaskBSASRec
    |
    +-- positive item score
    |
    +-- NegativeSamplerRec
            |
            v
       negative item score
            |
            v
       SASRecBPRLoss
```

For CE:

```text
TaskBSASRec
    |
    v
scores over all 165,469 candidate items
    |
    v
SASRecCELoss
```

The CE variant therefore does not require recommendation negative sampling for the loss itself.


# Parameter Ownership

The architecture deliberately separates shared and private parameters.

## Shared parameters

```text
SharedEmbedding
```

is updated by both Task A and Task B.


## Task A private parameters

```text
relation_embedding
```


## Task B private parameters

```text
position_embedding
Transformer encoder
LayerNorm
```

This design prevents separate item representations from being learned independently by the two tasks.


# SharedEmbedding Registration

`TaskATransE` and `TaskBSASRec` keep references to `SharedEmbedding` without registering it as their own child module.

This is intentional.

Otherwise, the same shared parameters could appear multiple times when constructing optimizers through the joint architecture.

`JointAlternateModelFashion` owns the `SharedEmbedding` directly.


# Main Differences from the B2B Architecture

Although the high-level alternate-learning design is shared with the B2B implementation, the Fashion version differs in several important aspects.

| Component | Fashion |
|---|---:|
| Shared embedding dimension | 64 |
| KG relations | 3 |
| KG entities | 182,983 |
| Shared rows including PAD | 182,984 |
| Unified padding ID | 182,983 |
| Item unified IDs | 17,514–182,982 |
| Recommendation items | 165,469 |
| SASRec layers | 4 |
| SASRec heads | 4 |
| Inner size | 256 |
| Max sequence length | 50 |
| Task A `compatible_with` | item → item |
| Entity normalization in Task A | enabled |
| Projection layer | none |

The lack of a projection layer is possible because:

```text
TransE dimension = SASRec hidden size = 64
```


# `compatible_with` Semantics

In the Fashion KG:

```text
compatible_with:
item -> item
```

Therefore, for type-constrained Task A evaluation:

```text
head candidates = items
tail candidates = items
```

This differs from the B2B domain, where the analogous structural relation connects different entity types.


# Smoke Tests

Each model module includes standalone smoke tests.

They are intended to verify structural behavior without requiring the complete alternate-learning training pipeline.


## `shared_embedding_fashion.py`

Checks include:

```text
matrix construction
padding row
gradient flow
TransE warm start
dimension mismatch detection
optional SASRec item warm start
diagnostic statistics
```


## `task_a_transe_fashion.py`

Checks include:

```text
TransE forward scoring
gradient flow
positive/negative scoring
relation warm start
parameter ownership
all-tail scoring
all-head scoring
score consistency
```


## `task_b_sasrec_fashion.py`

Checks include:

```text
SASRec forward pass
Fashion padding mask
gradient flow
candidate scoring
parameter ownership
legacy SASRec
RecBole warm start
```


## `joint_model_fashion.py`

Checks include:

```text
Task A forward
Task B forward
recommendation scoring
shared parameter ownership
shared object identity
Task A -> SharedEmbedding gradients
Task B -> SharedEmbedding gradients
padding behavior
```


## `losses_fashion.py`

Checks:

```text
TransE loss direction
BPR loss direction
CE loss direction
gradient propagation
integration with the joint model
```


## `samplers_fashion.py`

Checks:

```text
KGE corruption
head/tail distribution
valid entity range
recommendation negative sampling
positive-negative collision avoidance
padding exclusion
```


# Running the Smoke Tests

From the repository root:

```bash
python fashion_generalization/alternate_learning/models/shared_embedding_fashion.py

python fashion_generalization/alternate_learning/models/task_a_transe_fashion.py

python fashion_generalization/alternate_learning/models/task_b_sasrec_fashion.py

python fashion_generalization/alternate_learning/models/losses_fashion.py

python fashion_generalization/alternate_learning/models/samplers_fashion.py

python fashion_generalization/alternate_learning/models/joint_model_fashion.py
```


# Checkpoint Handling

The files in this directory do not define filesystem paths to `.pt` or `.pth` checkpoints.

Instead, pretrained tensors or state dictionaries are passed to the corresponding loading methods.

Physical checkpoint locations are defined by the configuration and execution layers.

For the standard Fashion setup, these are expected to be configured as:

```text
TransE:
results/hpo_transe/best_model.pt

SASRec:
results/sasrec_hpo/trial_017/best_valid.pth
```

The model classes therefore remain independent of machine-specific filesystem locations.


# Relationship with the Verification Checks

The integration of these model components is extensively verified by the scripts under:

```text
fashion_generalization/alternate_learning/checks/
```

Important checks include:

```text
verify_pretrained_weights_fashion.py
verify_shared_embedding_warmstart_fashion.py
verify_task_a_type_index_fashion.py
test_task_a_real_data_fashion.py
test_task_b_real_data_fashion.py
verify_warmstart_fashion.py
```

The smoke tests in this directory verify individual model components.

The scripts in `checks/` validate the same components using the real Fashion datasets and pretrained checkpoints.


# Final Model Structure

The final Fashion alternate-learning model can be summarized as:

```text
                       +----------------------+
                       | JointAlternateModel  |
                       |       Fashion        |
                       +----------+-----------+
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
             +-------------+             +-------------+
             |   Task A    |             |   Task B    |
             |   TransE    |             |   SASRec    |
             +------+------+             +------+------+
                    |                           |
                    |                           |
                    +------------+--------------+
                                 |
                                 v
                     +----------------------+
                     |   SharedEmbedding    |
                     |    182,984 x 64      |
                     +----------------------+
```

Task A provides structural KG supervision.

Task B provides sequential recommendation supervision.

Both tasks optimize the same shared item representations through alternate learning.
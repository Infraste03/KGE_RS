# Models

This directory contains the neural components used by the B2B alternate learning architecture.

The architecture alternately optimizes two tasks:

- Task A: Knowledge Graph link prediction using TransE
- Task B: sequential recommendation using SASRec

The two tasks are coupled through a single shared entity embedding matrix.

Both components operate in a 400-dimensional latent space, allowing the same item representations to be updated by both the knowledge graph and recommendation objectives. Although the two tasks are optimized separately, sequential recommendation in Task B is the primary target. Task A provides auxiliary Knowledge Graph supervision by updating the shared entity representations used by Task B.

## Architecture

The central component of the architecture is the `SharedEmbedding`.

Instead of maintaining separate item embeddings for TransE and SASRec, both tasks access the same embedding matrix.

During alternate learning:

1. Task A computes the TransE loss and updates the shared representations.
2. Task B computes the recommendation loss and updates the same representations.
3. The two objectives therefore progressively influence the same item embedding space.

The task-specific components remain separate:

- TransE has its own relation embeddings.
- SASRec has its own Transformer layers and positional embeddings.
- Entity and item representations are shared.

## Files

### `shared_embedding.py`

Defines the shared entity embedding matrix used by both Task A and Task B.

The matrix contains representations for all entities in the unified ID space, including:

- items
- machines
- product models
- clients

The embedding matrix is initialized from the pretrained TransE entity embeddings.

SASRec does not overwrite these item embeddings during initialization. Instead, the pretrained SASRec Transformer and positional parameters are loaded separately.

This initializes the shared representation space with the structural information learned by TransE while reusing the Transformer parameters learned by the standalone SASRec model.

The class also provides diagnostic utilities to inspect embedding norms, NaN/Inf values, and gradient behavior.


### `task_a_transe.py`

Implements Task A of the alternate learning architecture.

Task A performs Knowledge Graph link prediction using TransE on top of the shared entity embeddings.

The module contains:

- task-specific relation embeddings
- TransE triple scoring
- scoring of candidate heads and tails for link prediction
- utilities for loading pretrained PyKEEN relation embeddings
- diagnostic utilities

The entity embeddings are not owned by this module: they are retrieved directly from `SharedEmbedding`.

As a consequence, gradients produced by the TransE objective update the same entity representations that are used by SASRec.


### `task_b_sasrec.py`

Implements Task B of the alternate learning architecture.

Task B performs sequential recommendation using a SASRec architecture that reads item representations directly from `SharedEmbedding`.

The model uses:

- shared 400-dimensional item embeddings
- positional embeddings
- multi-head self-attention
- Transformer feed-forward layers
- causal attention masking

No projection layer is required because the SASRec hidden dimension and the TransE embedding dimension are both set to 400.

The pretrained standalone SASRec model is used to warm-start the Transformer, position embedding, and normalization parameters, while its original item embedding matrix is intentionally excluded.

The module also contains a legacy SASRec implementation used only for validation and reproducibility checks against the original standalone model.


### `joint_model.py`

Defines `JointAlternateModel`, which combines the shared embedding, Task A, and Task B into a single architecture.

It creates:

- one `SharedEmbedding`
- one `TaskATransE`
- one `TaskBSASRec`

and provides explicit methods for the two tasks:

- `forward_kge()` for Knowledge Graph scoring
- `forward_sasrec()` for sequential recommendation
- `score_all_tails_kge()` and `score_all_heads_kge()` for link prediction
- `score_items_sasrec()` for item ranking

The model also provides utilities for loading the pretrained TransE relation embeddings and pretrained SASRec Transformer parameters.

### `losses.py`

Defines the task-specific optimization objectives.

For Task A, `TransELoss` implements a margin-based ranking loss that encourages positive KG triples to obtain better TransE scores than corrupted negative triples.

For Task B, `SASRecBPRLoss` implements Bayesian Personalized Ranking loss, encouraging the recommendation score of the positive item to be higher than the score of sampled negative items.


### `samplers.py`

Defines the negative sampling strategies used during alternate learning.

`NegativeSamplerKGE` generates negative Knowledge Graph triples by randomly corrupting either the head or the tail entity.

`NegativeSamplerRec` generates negative recommendation items for Task B and supports sampling one or multiple negatives per positive item.

The recommendation sampler also handles the padding index used in the unified ID space.


## Warm Start

The alternate learning model is not initialized completely at random.

Task A is warm-started from the pretrained TransE model:

- entity embeddings initialize `SharedEmbedding`
- relation embeddings initialize the private TransE relation matrix

Task B is warm-started from the pretrained standalone SASRec model:

- positional embeddings are transferred
- Transformer parameters are transferred
- normalization parameters are transferred
- the original SASRec item embedding is not transferred

The shared item representation therefore starts from the Knowledge Graph embedding space and is subsequently updated by both tasks during alternate learning. This shared representation is the central mechanism through which Knowledge Graph supervision influences sequential recommendation in KGSEQ.
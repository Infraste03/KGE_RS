# Knowledge Graph and KGE Training

This directory contains the scripts used to construct and validate the B2B knowledge graph, train and optimize Knowledge Graph Embedding models, and evaluate their performance on the `compatible_with` link prediction task.

The Knowledge Graph component corresponds to Steps 1 and 2 of the proposed architecture:

1. Knowledge Graph construction and validation
2. Knowledge Graph Embedding training and model selection


## Knowledge Graph Construction

The graph is built exclusively from the training portion of the B2B dataset in order to avoid information leakage from the validation and test sets.

The graph contains four entity types:

- clients
- items
- machines
- product models

and five relation types:

- `owns`: client → machine
- `instance_of`: machine → product model
- `bought`: client → item
- `compatible_with_model`: item → product model
- `compatible_with`: item → machine

The `compatible_with` relation is particularly important because it is used as the prediction target for Task A.

The available `compatible_with` triples are deterministically split into:

- 90% training
- 5% validation
- 5% test

using random seed 42.

Validation and test `compatible_with` triples are excluded from the training graph and are used only for link prediction evaluation.


## KGE Models

Four Knowledge Graph Embedding models were considered:

- TransE
- RotatE
- DistMult
- ComplEx

The models represent different families of KGE approaches:

- **TransE**: translational model, based on the relation
  `h + r ≈ t`
- **RotatE**: represents relations as rotations in a complex-valued embedding space
- **DistMult**: bilinear model based on a diagonal relation matrix
- **ComplEx**: bilinear model operating in a complex-valued embedding space


## Hyperparameter Optimization

Hyperparameter optimization is performed using PyKEEN and Optuna.

The optimization objective is the Mean Reciprocal Rank (MRR) measured on the validation set.

The test set is never used during hyperparameter search and is evaluated only after the best configuration for each model has been selected.

The search uses the Tree-structured Parzen Estimator (TPE) sampler with up to 50 trials per model.

The optimized hyperparameters include:

- embedding dimension
- learning rate
- number of negative samples per positive triple
- margin of the Margin Ranking Loss
- regularization weight for models where explicit L2 regularization is used

The final training uses early stopping, with a maximum of 300 epochs.

The KGE evaluation focuses on link prediction for the `compatible_with` relation.


## Evaluation Protocol

Each KGE model is evaluated using filtered link prediction under two different settings.

### Naive evaluation

The correct entity is ranked against all entities contained in the Knowledge Graph.

This corresponds to the standard KGE evaluation setting.

### Type-constrained evaluation

The candidate set is restricted according to the semantic type required by the relation.

For `compatible_with`:

- tail prediction considers only machines
- head prediction considers only items

This evaluation is particularly relevant for the B2B domain, where the Knowledge Graph is heterogeneous and entities have clearly defined semantic types.


## KGE Results

The final results obtained after hyperparameter optimization are:

| Model | Evaluation | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---|---:|---:|---:|---:|
| RotatE | naive | 0.3147 | 0.2083 | 0.3503 | 0.5365 |
| RotatE | type-constrained | **0.3148** | **0.2083** | **0.3504** | **0.5367** |
| DistMult | naive | 0.2942 | 0.1912 | 0.3301 | 0.5037 |
| DistMult | type-constrained | 0.2973 | 0.1929 | 0.3335 | 0.5103 |
| TransE | naive | 0.2687 | 0.1554 | 0.3118 | 0.4936 |
| TransE | type-constrained | 0.2953 | 0.1922 | 0.3321 | 0.5017 |
| ComplEx | naive | 0.1724 | 0.0929 | 0.1842 | 0.3361 |
| ComplEx | type-constrained | 0.1948 | 0.1100 | 0.2105 | 0.3686 |

RotatE achieves the best overall performance, obtaining the highest MRR and Hits@10.

For RotatE, the results obtained with naive and type-constrained evaluation are almost identical, suggesting that the model is already able to strongly separate entity types in the learned embedding space.


## Why TransE Is Used in the Alternate Learning Architecture

Although RotatE achieves the best standalone KGE performance, it is not used as the Knowledge Graph backbone of the subsequent alternate learning architecture.

The final architecture requires the Knowledge Graph model and SASRec to share and jointly update item representations.

This introduces an architectural compatibility constraint.

RotatE operates in a **complex-valued embedding space**. Its representation therefore differs structurally from the standard real-valued embeddings used by SASRec.

TransE, instead, represents entities using standard **real-valued vectors**, making its embeddings directly compatible with the representation used by the sequential recommender.

For this reason, TransE was retained as the KGE backbone for the alternate learning stage even though its standalone link-prediction performance is lower than RotatE.

The choice is therefore not based exclusively on the standalone KGE ranking, but on the requirements of the subsequent joint architecture.

This choice is also aligned with the translational KGE formulations considered in alternate-learning approaches discussed by Zhang et al. (2024).

The selected TransE configuration uses an embedding dimension of **400**. The SASRec hidden dimension is consequently also fixed to 400 so that both components can operate on a dimensionally compatible shared representation.


## Files

### `build_kg.py`

Constructs the B2B Knowledge Graph from the training dataset.

The script:

- loads and validates the original training data
- checks missing and inconsistent values
- resolves machine-to-model inconsistencies using the most frequent association
- constructs the five KG relations
- applies the support threshold for `compatible_with_model`
- splits `compatible_with` triples into training, validation, and test sets
- prevents Task A validation/test leakage
- computes structural statistics of the resulting graph

The main generated files are:

`data/processed/kg_train.tsv`

`data/processed/taskA_valid.tsv`

`data/processed/taskA_test.tsv`

`data/processed/kg_stats.json`


### `validate_kg.py`

Performs an additional validation of the Knowledge Graph after construction.

The script performs two levels of verification.

The first consists of sampled spot checks, where triples from each relation are compared with the original interaction data.

The second checks global graph invariants, including:

- absence of duplicate triples
- consistency between relation types and entity prefixes
- uniqueness of the machine-to-model mapping
- absence of Task A leakage
- completeness of the `compatible_with` train/validation/test split


### `train_kge.py`

Provides the initial KGE training and evaluation pipeline used to compare the four candidate models under a common fixed configuration.

It trains:

- TransE
- RotatE
- DistMult
- ComplEx

and evaluates each model using both naive and type-constrained filtered link prediction.

This script was used for the initial KGE comparison and for validating the evaluation pipeline before the final hyperparameter optimization.


### `run_hpo_kge.py`

Performs the final hyperparameter optimization of the four KGE models using PyKEEN and Optuna.

For each model, the script:

- creates a resumable Optuna study
- optimizes MRR on the validation set
- applies early stopping
- supports an HPC time budget
- retrains the model using the best hyperparameters
- evaluates the selected model on the test set
- performs both naive and type-constrained evaluation
- saves the selected model weights and final metrics

The test set is not used during HPO.


### `crosscheck_eval.py`

Validates the correctness of the custom link-prediction evaluation implemented in `train_kge.py`.

The script compares the custom naive filtered evaluation with PyKEEN's standard `RankBasedEvaluator` on the same TransE model and test triples.

Its purpose is to verify that the custom computation of MRR, Mean Rank and Hits@K is consistent with the PyKEEN reference implementation.


### `KGE-TRAIN_HPO.sh`

SLURM submission script used to run the KGE hyperparameter optimization on the HPC infrastructure.

The script:

- requests a GPU node
- activates the Python virtual environment
- checks the required Python libraries
- checks CUDA availability
- verifies the presence of the KG input files
- launches `run_hpo_kge.py`
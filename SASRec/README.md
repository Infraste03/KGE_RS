# SASRec

This directory contains the scripts used to train, optimize, evaluate, and assess the stability of the SASRec sequential recommender used in the B2B knowledge-enhanced recommendation architecture.

The implementation is based on the SASRec model available in RecBole.

## Embedding Dimension Choice

SASRec represents both items and user interaction sequences in a latent embedding space.

In the proposed architecture, SASRec is subsequently combined with TransE through a shared embedding representation. For this reason, the dimensionality of the two embedding spaces must be consistent.

The TransE model selected in the Knowledge Graph Embedding stage uses an embedding dimension of 400. Therefore, the SASRec embedding dimension was fixed a priori to:

`hidden_size = 400`

rather than treating the embedding dimension as a hyperparameter during the final SASRec optimization.

The remaining SASRec hyperparameters were then optimized while keeping the embedding dimension fixed. This ensures that the resulting SASRec representation is dimensionally compatible with the TransE embeddings used in the subsequent joint architecture.

The hyperparameter search considered parameters such as:

- number of Transformer layers
- number of attention heads
- learning rate
- hidden dropout probability
- attention dropout probability
- weight decay
- loss function

## Final SASRec Performance

After hyperparameter optimization with the embedding dimension fixed to 400, the selected SASRec configuration was evaluated across multiple seeds.

The final performance used as the Step 3 SASRec baseline is:

| Model | Recall@20 | NDCG@20 |
|---|---:|---:|
| SASRec 400-dim | 0.3452 ± 0.0185 | 0.1840 ± 0.0074 |

These results represent the standalone SASRec performance before its integration with the Knowledge Graph component.

## Pretrained SASRec Weights

The pretrained SASRec weights used to initialize the sequential recommendation component during the alternate training stage are stored in:

`KG_RS/SASRec/pth/`

This checkpoint corresponds to the selected 400-dimensional SASRec model and is used as the pretrained initialization for Task B in the subsequent alternate training procedure.

## Files

### `run_hpo_sasrec400HPC.py`

Performs the hyperparameter optimization used to select the final SASRec configuration.

The embedding dimension is fixed to 400 to match the dimensionality of the TransE embeddings, while the remaining SASRec hyperparameters are optimized.

The script is designed to run on the HPC infrastructure.

### `run_hpo_sasrec.py`

Performs an earlier SASRec hyperparameter search using an embedding dimension of 512.

This script represents a preliminary experiment and is not the configuration used for the final Step 3 model.

### `embedding_dim_sasrec.py`

Loads a previously trained SASRec checkpoint and evaluates it on the B2B test set.

The main evaluation metrics are:

- Recall@20
- NDCG@20

### `run_stability_sasrec.py`

Performs the stability analysis of the selected SASRec configuration.

The same hyperparameter configuration is trained using different seeds in order to quantify the variability of the model performance.

The final results are reported as mean and standard deviation for:

- Recall@20
- NDCG@20

## `sbatch file/`

This directory contains the SLURM scripts used to execute the SASRec experiments on the HPC cluster.

### `submit_hpo_sasrec.sh`

Launches the preliminary SASRec hyperparameter optimization.

### `submit_hpo_400sasrec.sh`

Launches the final SASRec hyperparameter optimization with embedding dimension fixed to 400.

## Dataset

The experiments use the pre-split B2B-Parts-Rec interaction dataset with training, validation, and test sets.

The main evaluation metrics used throughout the experiments are Recall@20 and NDCG@20.
# Fashion Alternate Learning - Configuration Files

This directory contains the configuration files used for the Fashion alternate-learning experiments.

The configurations cover:

- standalone SASRec multi-seed retraining;
- local Step 4 alternate-learning experiments;
- HPC Step 4 experiments;
- BPR-based Task B training;
- CE-based Task B training;
- the separate Fashion-Heavy experimental setting.

All paths in the standard Fashion configurations are relative to:

```text
fashion_generalization/
```

The configuration files should therefore remain portable and should not contain machine-specific absolute paths.


# Directory Contents

```text
configs/
├── README.md
├── sasrec_fashion_seeds.yaml
├── step4_config_fashion.yaml
├── step4_config_fashion_v3.yaml
├── step4_hpc_config_fashion.yaml
└── step4_hpc_config_fashion_ce.yaml
```


# Standard Fashion Data

The standard Fashion experiments use the RecBole dataset located under:

```text
fashion_generalization/data/recbole/
```

with interaction file:

```text
data/recbole/fashion/fashion.inter
```

The Knowledge Graph files are:

```text
data/processed/kg_train.tsv
data/processed/taskA_valid.tsv
data/processed/taskA_test.tsv
```


# Pretrained Checkpoints

The standard alternate-learning experiments use two independently pretrained components.

## TransE

The pretrained TransE model is expected at:

```text
results/hpo_transe/best_model.pt
```

The Fashion TransE embedding dimensionality is:

```text
64
```


## SASRec

The pretrained SASRec checkpoint corresponding to the selected Fashion HPO trial is expected at:

```text
results/sasrec_hpo/trial_017/best_valid.pth
```

The selected standalone SASRec architecture uses:

```text
hidden_size = 64
n_layers = 4
n_heads = 4
inner_size = 256
hidden_dropout_prob = 0.5
attn_dropout_prob = 0.3
MAX_ITEM_LIST_LENGTH = 50
loss_type = CE
```


# `sasrec_fashion_seeds.yaml`

This configuration is used to retrain the selected standalone SASRec configuration across multiple random seeds.

It reproduces the hyperparameters selected by Fashion SASRec HPO trial 017.

Main configuration:

```text
hidden_size = 64
n_layers = 4
n_heads = 4
inner_size = 256

hidden_dropout_prob = 0.5
attn_dropout_prob = 0.3

MAX_ITEM_LIST_LENGTH = 50

loss_type = CE
learning_rate = 0.0001
weight_decay = 0.0001

train_batch_size = 256
eval_batch_size = 512

epochs = 200
stopping_step = 10
```

The default seed stored in the configuration is:

```text
2020
```

but the seed can be overridden by the execution script or SLURM job.

The evaluation protocol is:

```text
split = valid_and_test
order = TO
group_by = user
valid = full
test = full
```

with:

```text
Recall@20
NDCG@20
```

as evaluation metrics.


# `step4_config_fashion.yaml`

This is the standard local configuration for the Fashion Step 4 alternate-learning pipeline.

It combines:

```text
Task A = TransE
Task B = SASRec
```

through the shared entity embedding space.


## Pretrained initialization

Task A starts from:

```text
results/hpo_transe/best_model.pt
```

Task B Transformer components start from:

```text
results/sasrec_hpo/trial_017/best_valid.pth
```


## Architecture

```text
KGE / shared embedding dimension = 64

SASRec:
n_layers = 4
n_heads = 4
inner_size = 256
hidden_dropout = 0.5
attn_dropout = 0.3
```


## Alternate-learning settings

The configuration defines:

```text
batch_size_kge = 1024
batch_size_sasrec = 256
margin_loss = 8.684
learning_rate = 0.001
epochs = 2
eval_step = 1
num_neg_b = 1
```

The default scheduling strategy is:

```text
one_to_one
```

and may be overridden by the execution script.


## Task B objective

This configuration uses:

```text
loss_type = BPR
```

for Task B.

It is therefore the BPR version of the standard Fashion alternate-learning configuration.


# `step4_hpc_config_fashion.yaml`

This is the base configuration used for Fashion Step 4 experiments on the HPC environment.

Its architecture is equivalent to the standard Fashion configuration:

```text
KGE dimension = 64
SASRec layers = 4
SASRec heads = 4
inner size = 256
hidden dropout = 0.5
attention dropout = 0.3
```

The main difference from the small local configuration is the training horizon:

```text
epochs = 50
```

The Task B objective is:

```text
BPR
```

and the default scheduling strategy is:

```text
one_to_one
```

The configuration is intended to be consumed by the corresponding HPC training and HPO scripts.


# `step4_hpc_config_fashion_ce.yaml`

This configuration is the CE counterpart of the HPC Fashion alternate-learning configuration.

The architecture and pretrained initialization are the same as in the standard Fashion setup.

The important difference is the Task B objective:

```text
loss_type = CE
```

with:

```text
train_neg_sample_args = null
```

This configuration is used to evaluate how replacing the BPR objective with full categorical cross-entropy affects recommendation performance during alternate learning.


## BPR vs CE

The two HPC configurations allow a controlled comparison:

```text
step4_hpc_config_fashion.yaml
    -> Task B uses BPR

step4_hpc_config_fashion_ce.yaml
    -> Task B uses CE
```

The shared TransE initialization and SASRec architecture remain otherwise aligned.


# `step4_config_fashion_v3.yaml`

This configuration belongs to the separate Fashion-Heavy experimental setting.

It should not be interpreted as the same experiment as the standard 64-dimensional Fashion configuration.


## Dataset

The v3 RecBole dataset is expected under:

```text
data/recbole_v3/
```

with:

```text
data/recbole_v3/fashion_v3/fashion_v3.inter
```


## SASRec checkpoint

The selected v3 SASRec checkpoint is:

```text
results/sasrec_hpo_v3/trial_003/best_valid.pth
```


## Architecture

The v3 configuration uses a larger shared latent space:

```text
hidden_size = 128
embedding_size = 128
kge_dim = 128
```

with:

```text
n_layers = 2
n_heads = 8
inner_size = 256

hidden_dropout = 0.543705
attn_dropout = 0.216936
```

The Task B loss is:

```text
CE
```


## Alternate-learning settings

```text
scheduling = one_to_one
epochs = 50
eval_step = 1

learning_rate = 0.000172
batch_size_kge = 1024
batch_size_sasrec = 512
margin_loss = 7.379029
```

The v3 configuration must remain separate from the standard Fashion configuration because it uses a different dataset, pretrained SASRec checkpoint, and embedding dimensionality.


# Evaluation Protocol

The Fashion configurations use the RecBole sequential evaluation protocol:

```yaml
eval_args:
  split:
    LS: valid_and_test
  order: TO
  group_by: user
  mode:
    valid: full
    test: full
```

The evaluation metrics are:

```text
Recall@20
NDCG@20
```

with:

```text
NDCG@20
```

used as the primary validation metric.


# Relationship with the Pretraining Stages

The alternate-learning configuration assumes that the two model components have already been pretrained independently.

The pipeline is:

```text
TransE HPO
    |
    v
best_model.pt
    |
    v
SharedEmbedding + Task A
```

and:

```text
SASRec HPO
    |
    v
best_valid.pth
    |
    v
Task B Transformer warm start
```

The two pretrained components are then combined in the joint alternate-learning model.


# BPR and CE Experiments

The Fashion experiments include both BPR and CE variants for Task B.

The BPR configuration uses:

```text
loss_type = BPR
```

while the CE configuration uses:

```text
loss_type = CE
train_neg_sample_args = null
```

These configurations should be treated as separate experimental variants.

They allow the effect of the Task B optimization objective to be evaluated while keeping the main architecture and pretrained initialization fixed.


# Path Convention

The configuration files intentionally avoid absolute machine-specific paths.

For the standard Fashion experiment, paths such as:

```text
data/processed/kg_train.tsv
results/hpo_transe/best_model.pt
results/sasrec_hpo/trial_017/best_valid.pth
```

are interpreted relative to:

```text
fashion_generalization/
```

Execution scripts should therefore resolve the configured paths consistently from the Fashion project root.

HPC execution scripts may contain cluster-specific paths required to enter the correct working directory, but the YAML configuration itself should remain portable whenever possible.


# Verification

Before running Step 4 experiments, the pretrained checkpoints, mappings, and evaluation configuration can be verified using the scripts under:

```text
fashion_generalization/alternate_learning/checks/
```

In particular:

```text
verify_pretrained_weights_fashion.py
verify_id_alignment_fashion.py
verify_transe_metrics_fashion.py
verify_sasrec_metrics_fashion.py
verify_shared_embedding_warmstart_fashion.py
verify_warmstart_fashion.py
```

These checks verify that the configuration points to compatible pretrained models and that the integrated architecture can be initialized correctly before alternate learning starts.
# Models

This directory contains the model-specific training and hyperparameter optimization code used in the Fashion experiments.

```text
models/
├── sasrec/
└── TransE/
```

## `sasrec/`

Contains the SASRec training and hyperparameter optimization scripts used for the sequential recommendation task.

It includes the standard Fashion experiments, the Fashion v3 experiments, and a local smoke-test script.

## `TransE/`

Contains the TransE training, hyperparameter optimization, ablation, and Fashion v3 scripts used for the knowledge graph component.

This directory also contains the `3rel/` and `5rel/` Fashion v3 experiment outputs and pretrained TransE checkpoints.

See the README files inside each model directory for further details.
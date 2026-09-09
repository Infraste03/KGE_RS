# HPC Scripts

This directory contains the SLURM scripts used to run the Fashion experiments
on the HPC cluster.

The scripts cover different stages and experimental configurations, including:

- SASRec hyperparameter optimization
- SASRec multi-seed retraining
- TransE hyperparameter optimization
- Knowledge Graph ablation experiments
- Fashion v2 and v3 experiments
- 3-relation and 5-relation configurations
- Alternate-learning experiments

Some scripts correspond to specific experimental variants and were retained to
preserve the reproducibility of previously executed runs.

## HPC Paths

To keep the public repository portable and independent of a specific user
account or filesystem layout, user-specific absolute paths have been replaced
with the generic placeholder:


```text
/path/to/KG_RS
```

Python environments, SLURM partitions, memory requirements, GPU requests, and
time limits may also need to be adapted to the target cluster.

Before running any SLURM script, replace this placeholder with the absolute
path to the repository on the target HPC system.

## Source Code

The Python scripts launched by these SLURM files are located in the
corresponding directories under `fashion_generalization/`, including:

```text
models/
preprocessing/
alternate_learning/
alternate_learning/variants/
```

Experiment outputs and checkpoints are stored separately from the SLURM
scripts.
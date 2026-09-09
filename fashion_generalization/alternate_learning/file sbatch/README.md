# Fashion Alternate Learning - HPC / SLURM Scripts

This directory contains the SLURM submission scripts used to run the Fashion experiments on the University of Parma HPC infrastructure.

The scripts cover:

- standalone SASRec multi-seed retraining;
- Step 4 alternate-learning HPO;
- BPR-based Task B experiments;
- CE-based Task B experiments;
- extended HPO searches for the `one_to_one` and `epoch` scheduling strategies.

These scripts are designed for execution on an HPC cluster.

To keep the repository portable and independent of a specific user account or filesystem layout, user-specific absolute paths have been replaced with the placeholder:

    /path/to/KG_RS

Before running any script, replace this placeholder with the absolute path to the repository on the target cluster.


# Directory Contents

```text
hpc/
├── README.md
├── sbatch_sasrec_fashion_seeds.sh
├── sbatch_hpo_step4_fashion.sh
├── sbatch_hpo_step4_fashion_ce.sh
├── ext_epoch_sbatch_hpo_step4_fashion.sh
└── ext_one_sbatch_hpo_step4_fashion.sh
```


# HPC Environment

The scripts were designed for the project located under:

```text
/path/to/KG_RS/
```

`/path/to/KG_RS` is a placeholder and must be replaced with the actual absolute path to the repository on the target system.

For example:

    /hpc/scratch/<username>/KG_RS

This convention avoids embedding personal usernames or institution-specific paths in the public repository and allows the same scripts to be adapted to different HPC environments.

The Python virtual environment is expected at:

```text
/path/to/KG_RS/.venv/
```

The Fashion project is located at:

```text
/path/to/KG_RS/fashion_generalization/
```

The scripts load:

```text
gnu8/8.3.0
python/3.9.10
```

and request a GPU through SLURM.


# SLURM Resources

The scripts generally request:

```text
CPUs              = 8
Nodes             = 1
GPUs              = 1
Memory            = 50 GB
Maximum wall time = 23:59 hours
Partition         = gpu
QoS               = gpu
```

The corresponding SLURM directives are:

```bash
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=50G
#SBATCH --time=0-23:59:0
```


# Working Directory Convention

Some scripts use absolute HPC paths, while others invoke Python runners and configuration files using paths relative to the repository root.

For this reason, the recommended convention is to submit the jobs from:

```text
/path/to/KG_RS/
```

For example:

```bash
cd /path/to/KG_RS
```

before running:

```bash
sbatch fashion_generalization/alternate_learning/hpc/<script>.sh
```

This is particularly important for scripts that use paths such as:

```text
.venv/bin/activate

fashion_generalization/alternate_learning/run_hpo_step4_fashion.py

fashion_generalization/alternate_learning/configs/step4_hpc_config_fashion.yaml
```


# Checkpoint Paths

The SLURM scripts do not directly define the pretrained TransE or SASRec checkpoint paths.

Checkpoint locations are defined in the corresponding YAML configuration files under:

```text
fashion_generalization/alternate_learning/configs/
```

For the standard Fashion experiment, the expected pretrained models are:

```text
TransE:
results/hpo_transe/best_model.pt

SASRec:
results/sasrec_hpo/trial_017/best_valid.pth
```

Therefore, changes to checkpoint organization should normally be made in the YAML configuration files rather than in the SLURM scripts.


# `sbatch_sasrec_fashion_seeds.sh`

This script retrains the selected standalone SASRec configuration across multiple random seeds.

Each SLURM job runs exactly one seed.

The configuration is read from:

```text
fashion_generalization/alternate_learning/configs/sasrec_fashion_seeds.yaml
```

The RecBole dataset is read from:

```text
fashion_generalization/data/recbole/
```


## Usage

Submit one job for each seed:

```bash
sbatch fashion_generalization/alternate_learning/hpc/sbatch_sasrec_fashion_seeds.sh 2020

sbatch fashion_generalization/alternate_learning/hpc/sbatch_sasrec_fashion_seeds.sh 42

sbatch fashion_generalization/alternate_learning/hpc/sbatch_sasrec_fashion_seeds.sh 1024

sbatch fashion_generalization/alternate_learning/hpc/sbatch_sasrec_fashion_seeds.sh 999

sbatch fashion_generalization/alternate_learning/hpc/sbatch_sasrec_fashion_seeds.sh 2024
```

The seed passed as the first command-line argument overrides the default seed stored in the YAML configuration.


## Purpose

The resulting independent runs are used to estimate the variability of the standalone SASRec baseline and to support paired statistical comparisons with the alternate-learning models.


# `sbatch_hpo_step4_fashion.sh`

This is the main Step 4 HPO submission script for the standard Fashion alternate-learning implementation.

It supports two scheduling strategies:

```text
one_to_one
epoch
```


## Usage

```bash
sbatch fashion_generalization/alternate_learning/hpc/sbatch_hpo_step4_fashion.sh one_to_one
```

or:

```bash
sbatch fashion_generalization/alternate_learning/hpc/sbatch_hpo_step4_fashion.sh epoch
```

If no argument is provided, the default is:

```text
one_to_one
```


## Runner

The script executes:

```text
fashion_generalization/alternate_learning/run_hpo_step4_fashion.py
```

using:

```text
fashion_generalization/alternate_learning/configs/step4_hpc_config_fashion.yaml
```


## Study Names

The generated Optuna study names follow:

```text
step4_fashion_one_to_one
step4_fashion_epoch
```


# `sbatch_hpo_step4_fashion_ce.sh`

This script runs the CE version of the Fashion Step 4 HPO.

It uses:

```text
run_hpo_step4_fashion_ce.py
```

and:

```text
configs/step4_hpc_config_fashion_ce.yaml
```


## Usage

```bash
sbatch fashion_generalization/alternate_learning/hpc/sbatch_hpo_step4_fashion_ce.sh one_to_one
```

or:

```bash
sbatch fashion_generalization/alternate_learning/hpc/sbatch_hpo_step4_fashion_ce.sh epoch
```


## Task B Objective

The corresponding configuration uses:

```text
loss_type = CE
```

rather than BPR.

This allows a controlled comparison between the two recommendation objectives while maintaining the same general alternate-learning architecture.


## Study Names

The generated studies follow:

```text
step4_fashion_one_to_one_ce
step4_fashion_epoch_ce
```


# `ext_epoch_sbatch_hpo_step4_fashion.sh`

This script runs the extended HPO search dedicated to the:

```text
epoch
```

scheduling strategy.

The scheduling strategy is fixed inside the script:

```bash
SCHEDULING="epoch"
```

The study name is:

```text
step4_fashion_epoch_ext_v1
```

and the Python runner is:

```text
ext_epoch_run_hpo_step4_fashion.py
```


## Usage

```bash
sbatch fashion_generalization/alternate_learning/hpc/ext_epoch_sbatch_hpo_step4_fashion.sh
```

No scheduling argument is required because the strategy is fixed by the script.


# `ext_one_sbatch_hpo_step4_fashion.sh`

This script runs the extended HPO search dedicated to:

```text
one_to_one
```

scheduling.

The scheduling strategy is fixed as:

```bash
SCHEDULING="one_to_one"
```

The study name is:

```text
step4_fashion_one_to_one_ext_v1
```

and the runner is:

```text
ext_one_run_hpo_step4_fashion.py
```


## Usage

```bash
sbatch fashion_generalization/alternate_learning/hpc/ext_one_sbatch_hpo_step4_fashion.sh
```


# HPO Trial Timeout

The Step 4 HPO scripts support the optional environment variable:

```text
HPO_TRIAL_TIMEOUT
```

When defined, it is passed to the HPO runner as:

```text
--timeout
```

For example:

```bash
export HPO_TRIAL_TIMEOUT=21600
```

followed by the desired `sbatch` command.

If the variable is not defined, no explicit per-trial timeout argument is added.


# Output and Logs

SLURM logs are generated using the names defined in the corresponding `#SBATCH --output` directives.

Examples include:

```text
SASREC_Fashion_*.log
HPO_Fashion_*.log
HPO_CE_Fashion_*.log
HPO_ext_v1_epoch_*.log
HPO_ext_v1_one_to_one_*.log
```

The HPO scripts additionally create Optuna study directories for their respective experiments.


# BPR and CE HPO

The standard Fashion alternate-learning experiments distinguish between two Task B objectives:

```text
BPR
CE
```

The BPR HPO uses:

```text
sbatch_hpo_step4_fashion.sh
step4_hpc_config_fashion.yaml
run_hpo_step4_fashion.py
```

The CE HPO uses:

```text
sbatch_hpo_step4_fashion_ce.sh
step4_hpc_config_fashion_ce.yaml
run_hpo_step4_fashion_ce.py
```

These experiments should be treated as separate HPO studies.


# Extended HPO

The extended HPO scripts were introduced to explore larger or revised search spaces separately for the two scheduling strategies.

They use dedicated Python runners:

```text
ext_epoch_run_hpo_step4_fashion.py
ext_one_run_hpo_step4_fashion.py
```

and separate study names to avoid overwriting the original HPO studies.


# Portability
The `.sh` files preserve the SLURM configuration and execution logic used for the experiments, while user-specific filesystem paths have been replaced with the generic `/path/to/KG_RS` placeholder.

When running the repository on another cluster, the following may need to be adapted:

```text
project root
virtual environment path
module versions
SLURM partition
SLURM QoS
GPU resource syntax
memory and time limits
```

The Python source code and YAML configurations should remain as portable as possible, while these SLURM scripts act as environment-specific launchers.


# Before Running

Before submitting expensive HPC experiments, verify the Fashion pipeline using the checks under:

```text
fashion_generalization/alternate_learning/checks/
```

In particular, the pretrained checkpoints and warm-start integration should be validated before launching Step 4 HPO.

The most comprehensive integration check is:

```text
verify_warmstart_fashion.py
```

A successful verification confirms that the TransE and SASRec components can be loaded consistently into the Fashion joint model before alternate-learning training begins.
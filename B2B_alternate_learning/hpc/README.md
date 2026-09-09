# HPC / SLURM Execution Scripts

This directory contains the SLURM submission scripts used to execute the B2B
recommendation experiments on an HPC cluster.

The scripts do not implement the models themselves. They are execution wrappers
responsible for:

- requesting HPC resources;
- loading the required software environment;
- activating the Python virtual environment;
- selecting experiment-specific parameters;
- launching the corresponding Python scripts;
- assigning random seeds or ablation variants;
- organizing SLURM logs and experiment outputs.

The main Python implementation is located under:

```text
B2B_alternate_learning/
```

## Directory Scope

The HPC directory contains launchers for:

```text
Step 4 hyperparameter optimization
standard alternate-learning training
final multi-seed experiments
standalone SASRec multi-seed retraining
Knowledge Graph ablation experiments
multi-seed ablation experiments
```

The current scripts are:

```text
hpc/
├── README.md
├── sbatch_hpo_step4.sh
├── sbatch_one_seed.sh
├── sbatch_sasrec_seeds.sh
├── sbatch_seeds_step4_epoch.sh
├── sbatch_step4_abaltion.sh
├── sbatch_step4_ablation_seeds_v1.sh
├── sbatch_step4_ablation_seeds_v2.sh
├── sbatch_step4_ablation_v2.sh
└── sbatch_step4_alternate.sh
```

## Portability

The public repository does not rely on user-specific usernames or personal
filesystem locations.

Paths used by the SLURM scripts should be interpreted relative to the repository
organization whenever possible.

Before running the scripts on another HPC system, verify the local cluster
configuration, including:

```text
repository location
virtual-environment location
dataset paths
configuration-file paths
checkpoint paths
output directories
SLURM partition
SLURM QoS
software modules
GPU availability
```

A typical repository location on an HPC system could be:

```text
/hpc/scratch/<username>/KG_RS
```

but the actual path depends on the target infrastructure.

## Common SLURM Configuration

Most experiments were executed with approximately the following resources:

```text
Nodes:           1
GPUs:            1
CPUs per task:   8
Memory:          50 GB
Partition:       gpu
QoS:             gpu
Maximum runtime: approximately 24 hours
```

The software environment typically includes:

```bash
module purge
module load gnu8/8.3.0
module load python/3.9.10
```

followed by activation of the project virtual environment.

Most scripts also execute:

```bash
nvidia-smi
```

before training to verify that the allocated GPU is visible.

These settings reflect the environment used for the original experiments and
may need to be adapted to another cluster.

## Script Overview

| Script | Purpose |
|---|---|
| `sbatch_hpo_step4.sh` | Step 4 hyperparameter optimization |
| `sbatch_step4_alternate.sh` | Standard Step 4 alternate-learning training |
| `sbatch_one_seed.sh` | Final `one_to_one` Step 4 run for one random seed |
| `sbatch_seeds_step4_epoch.sh` | Final `epoch` Step 4 run for one random seed |
| `sbatch_sasrec_seeds.sh` | Standalone SASRec multi-seed retraining |
| `sbatch_step4_abaltion.sh` | Original Step 4 KG ablation launcher |
| `sbatch_step4_ablation_v2.sh` | Corrected Step 4 KG ablation launcher |
| `sbatch_step4_ablation_seeds_v1.sh` | Additional seeds for the first group of ablation variants |
| `sbatch_step4_ablation_seeds_v2.sh` | Additional seeds for the second group of ablation variants |

## Step 4 Hyperparameter Optimization

### `sbatch_hpo_step4.sh`

This script launches hyperparameter optimization for the Step 4
alternate-learning architecture.

The scheduling strategy is provided as a command-line argument.

Supported strategies are:

```text
one_to_one
epoch
adaptive
```

Example:

```bash
sbatch sbatch_hpo_step4.sh one_to_one
```

The selected strategy is used to define the corresponding Optuna study and
output location.

The script launches:

```text
B2B_alternate_learning/run_hpo_step4.py
```

and passes the scheduling-specific parameters required by the HPO pipeline.

An optional trial timeout can be controlled through:

```text
HPO_TRIAL_TIMEOUT
```

when supported by the execution environment.

This script represents the main HPC launcher used during Step 4 hyperparameter
search.

## Standard Alternate-Learning Training

### `sbatch_step4_alternate.sh`

This script launches a standard Step 4 alternate-learning experiment after
selecting a scheduling strategy.

Example:

```bash
sbatch sbatch_step4_alternate.sh one_to_one
```

Supported strategies are:

```text
one_to_one
epoch
adaptive
```

The script launches:

```text
B2B_alternate_learning/run_step4.py
```

using the corresponding Step 4 configuration.

Its purpose is to train the alternate-learning architecture without performing
a new hyperparameter search.

## Final Multi-Seed Step 4 Experiments

After selecting the final hyperparameters, the model is retrained with multiple
random seeds to estimate performance stability.

The final seed set used in the B2B experiments is:

```text
2020
42
999
1024
2024
```

### `sbatch_one_seed.sh`

This script launches one final Step 4 run using:

```text
scheduling = one_to_one
```

The seed is passed as a command-line argument.

Example:

```bash
sbatch sbatch_one_seed.sh 2020
```

The script uses:

```text
B2B_alternate_learning/configs/step4_seeds_one_to_one.yaml
```

and launches:

```text
B2B_alternate_learning/run_step4.py
```

with the selected seed and a seed-specific run tag.

The run tag ensures that results from different random seeds are stored
separately.

### `sbatch_seeds_step4_epoch.sh`

This script performs the corresponding final multi-seed evaluation for:

```text
scheduling = epoch
```

Example:

```bash
sbatch sbatch_seeds_step4_epoch.sh 42
```

It uses:

```text
B2B_alternate_learning/configs/step4_seeds_epoch.yaml
```

and launches:

```text
B2B_alternate_learning/run_step4.py
```

with the selected seed.

Together, these scripts provide the independent runs required to report:

```text
mean Recall@20
standard deviation of Recall@20
mean NDCG@20
standard deviation of NDCG@20
```

for the final Step 4 models.

## Standalone SASRec Multi-Seed Baseline

### `sbatch_sasrec_seeds.sh`

This script retrains the standalone SASRec baseline using independent random
seeds.

The seed is supplied at submission time.

Example:

```bash
sbatch sbatch_sasrec_seeds.sh 2020
```

The script uses:

```text
B2B_alternate_learning/configs/sasrec400_seeds.yaml
```

and executes the SASRec training through RecBole.

The resulting independent runs are used to compute the mean and standard
deviation of:

```text
Recall@20
NDCG@20
```

and to support paired statistical comparisons with the alternate-learning
models.

When performing paired statistical tests, the compared configurations must use
the same set of random seeds.

## Knowledge Graph Ablation Experiments

The Step 4 ablation experiments measure how the relation composition of the
Knowledge Graph affects downstream recommendation performance.

Five selected KG configurations are propagated from the isolated TransE
ablation study to Step 4:

```text
loo_no_compatible_with
loo_no_instance_of
mix1_keep_compatible_with_owns
mix3_keep_bought_compatible_with
targeted_triple_cw_bought_instance
```

The corresponding Stage 1 TransE ablation experiments and their results are
documented under:

```text
ablation_study/
```

The Step 4 ablation experiments use:

```text
scheduling = one_to_one
```

## Original Step 4 Ablation Launcher

### `sbatch_step4_abaltion.sh`

This script belongs to the first implementation of the Step 4 ablation
experiments.

The filename intentionally preserves the historical typo:

```text
abaltion
```

The script defines a SLURM job array covering the five selected KG variants:

```bash
#SBATCH --array=0-4
```

Each array task is mapped to one ablation configuration.

The script launches:

```text
B2B_alternate_learning/run_step4_ablation.py
```

using:

```text
B2B_alternate_learning/configs/step4_seeds_one_to_one.yaml
```

This launcher is retained because part of the final ablation experiment set was
executed through this pipeline.

## Corrected Step 4 Ablation Launcher

### `sbatch_step4_ablation_v2.sh`

This script was introduced for the corrected execution of selected Step 4
ablation experiments.

It covers the same five KG configurations:

```text
loo_no_compatible_with
loo_no_instance_of
mix1_keep_compatible_with_owns
mix3_keep_bought_compatible_with
targeted_triple_cw_bought_instance
```

through a SLURM job array.

A subset of array indices can also be selected directly during submission.

For example:

```bash
sbatch --array=1-3 sbatch_step4_ablation_v2.sh
```

This was useful when only specific configurations required rerunning after
corrections to the ablation pipeline.

## Multi-Seed Ablation Experiments

Additional random seeds were executed after the initial Step 4 ablation runs.

The additional seeds are:

```text
42
1024
999
2024
```

The original run provides the remaining seed used in the final five-run
evaluation.

The multi-seed executions are divided into two SLURM scripts.

### `sbatch_step4_ablation_seeds_v1.sh`

This script runs additional seeds for:

```text
loo_no_compatible_with
targeted_triple_cw_bought_instance
```

The experiment grid contains:

```text
2 variants × 4 additional seeds = 8 jobs
```

and therefore uses:

```bash
#SBATCH --array=0-7
```

Each task receives:

```text
variant
seed
run tag
```

and launches the corresponding Step 4 ablation training.

### `sbatch_step4_ablation_seeds_v2.sh`

This script runs additional seeds for:

```text
loo_no_instance_of
mix1_keep_compatible_with_owns
mix3_keep_bought_compatible_with
```

The grid contains:

```text
3 variants × 4 additional seeds = 12 jobs
```

and therefore uses:

```bash
#SBATCH --array=0-11
```

As in the first multi-seed script, each run receives a variant, random seed,
and seed-specific run tag.

Together:

```text
sbatch_step4_ablation_seeds_v1.sh
sbatch_step4_ablation_seeds_v2.sh
```

cover the additional random-seed experiments for all five selected B2B
ablation variants.

## Experimental Workflow

The main alternate-learning workflow represented by these scripts is:

```text
Step 4 HPO
    |
    v
sbatch_hpo_step4.sh
    |
    v
Select scheduling strategy and hyperparameters
    |
    v
Standard Step 4 training
    |
    v
sbatch_step4_alternate.sh
    |
    v
Final multi-seed evaluation
    |
    +-----------------------------+
    |                             |
    v                             v
sbatch_one_seed.sh       sbatch_seeds_step4_epoch.sh
    |
    v
Comparison with standalone SASRec
    |
    v
sbatch_sasrec_seeds.sh
```

The Knowledge Graph ablation workflow is:

```text
Isolated TransE KG ablation
    |
    v
Selection of relevant KG configurations
    |
    v
Initial Step 4 ablation runs
    |
    +-----------------------------+
    |                             |
    v                             v
Original launcher             Corrected launcher
    |                             |
    v                             v
sbatch_step4_abaltion.sh     sbatch_step4_ablation_v2.sh
    |
    v
Additional multi-seed evaluation
    |
    +----------------------------------------+
    |                                        |
    v                                        v
sbatch_step4_ablation_              sbatch_step4_ablation_
seeds_v1.sh                         seeds_v2.sh
```

## Logs and Outputs

Each SLURM script generates its own log files.

Typical patterns include:

```text
HPO_*.log
Step4_*.log
SEED_*.log
SASREC_*.log
Step4_Ablation_*.log
Step4_Abl_Seed_*.log
```

The exact log location depends on the directory from which the SLURM job is
submitted and on the corresponding `#SBATCH --output` directive.

Generated artifacts can include:

```text
model checkpoints
training histories
evaluation metrics
Optuna studies
SLURM logs
```

Large generated artifacts are stored separately from the source code whenever
possible.

The main repository-level experiment outputs are organized under:

```text
results/
```

while the consolidated ablation outputs are documented under:

```text
ablation_study/
```

## Before Running on Another HPC System

Before submitting one of these scripts on another cluster, verify:

1. the repository location;
2. the virtual-environment location;
3. the Python script referenced by the launcher;
4. the configuration-file path;
5. the dataset location;
6. the pretrained checkpoint locations;
7. the output directories;
8. the SLURM partition and QoS;
9. the available software modules;
10. the requested GPU, CPU, memory, and runtime resources;
11. the random seed or ablation variant required by the experiment.

Cluster-specific settings should be adapted to the target infrastructure
without introducing user-specific paths into the public repository.

## Scope of This Directory

The files in this directory are execution wrappers.

They do not define:

```text
model architecture
loss functions
sampling strategies
evaluation metrics
unified ID mappings
alternate-learning logic
```

Those components are implemented in the Python source files under:

```text
B2B_alternate_learning/
```

The purpose of this directory is to document how the different B2B experiments
were submitted and executed on HPC infrastructure.
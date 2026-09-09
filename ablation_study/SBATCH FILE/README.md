# SLURM Scripts for Ablation Experiments

This directory contains the SLURM job-array scripts used to execute the
Knowledge Graph ablation experiments on an HPC cluster.

## Scripts

```text
SBATCH FILE/
├── README.md
├── sbatch_transe_ablation.sh
├── sbatch_v3_ablation_3rel.sh
└── sbatch_v3_ablation_5rel.sh
```

### `sbatch_transe_ablation.sh`

Runs the **B2B isolated TransE ablation study**.

The job array contains 13 ablation variants:

```text
5 leave-one-out
3 random-mix
5 targeted
```

The corresponding Python runner is:

```text
ablation_study/run_transe_ablation_batch.py
```

### `sbatch_v3_ablation_3rel.sh`

Runs the **Fashion v3 3-relation TransE ablation study**.

The job array contains 4 variants:

```text
3 leave-one-out
1 targeted
```

The corresponding runner is:

```text
ablation_study/fashionv3/run_v3_ablation_3rel.py
```

### `sbatch_v3_ablation_5rel.sh`

Runs the **Fashion v3 5-relation TransE ablation study**.

The job array contains 13 variants:

```text
5 leave-one-out
3 random-mix
5 targeted
```

The corresponding runner is:

```text
ablation_study/fashionv3/run_v3_ablation_5rel.py
```

## Repository Path

To keep the public repository portable and independent of a specific user or
HPC filesystem layout, the scripts use the placeholder:

```text
/path/to/KG_RS
```

Before submitting a job, replace this value with the absolute path to the
repository on the target cluster.

For example:

```text
/hpc/scratch/<username>/KG_RS
```

The scripts assume that the Python virtual environment is located at:

```text
/path/to/KG_RS/.venv/
```

## HPC Configuration

The scripts were configured with the following SLURM resources:

```text
CPUs              = 8
Nodes             = 1
GPUs              = 1
Memory            = 50 GB
Maximum wall time = 23:59 hours
Partition         = gpu
QoS               = gpu
```

The module names and versions used in the scripts may need to be adapted to
the target HPC environment.

## Usage

After setting `PROJECT_ROOT` in the desired script, submit the job with:

```bash
sbatch <script_name>.sh
```

No variant argument is required because each script uses a SLURM job array to
assign one ablation variant to each array task.

## Outputs

SLURM log files are written using the corresponding `#SBATCH --output`
configuration.

Experimental models, metrics, and configuration files are produced by the
Python runners and stored in their respective result directories under
`ablation_study/`.
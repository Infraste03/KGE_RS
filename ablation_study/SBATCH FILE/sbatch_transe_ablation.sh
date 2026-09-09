#!/bin/bash
# ==============================================================================
# SLURM SUBMISSION SCRIPT — B2B TRANSE ABLATION (JOB ARRAY)
# ==============================================================================
# Usage:
#   sbatch sbatch_transe_ablation.sh
#
# No command-line arguments are required.
# The SLURM job array executes all 13 ablation variants.
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. SLURM settings
# ------------------------------------------------------------------------------

#SBATCH --job-name=TransE_Ablation
#SBATCH --output=Ablation_%x_%A_%a.log
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=50G
#SBATCH --time=0-23:59:0
#SBATCH --array=0-12


# ------------------------------------------------------------------------------
# 2. Repository path
# ------------------------------------------------------------------------------

# Replace this placeholder with the absolute path to the repository
# on the target HPC system.
PROJECT_ROOT="/path/to/KG_RS"


# ------------------------------------------------------------------------------
# 3. Map array index to ablation variant
# ------------------------------------------------------------------------------

VARIANTS=(
    "loo_no_bought"
    "loo_no_compatible_with"
    "loo_no_compatible_with_model"
    "loo_no_instance_of"
    "loo_no_owns"
    "mix1_keep_compatible_with_owns"
    "mix2_keep_bought_owns"
    "mix3_keep_bought_compatible_with"
    "targeted_pair_cw_cwm"
    "targeted_pair_cw_instance"
    "targeted_single_cw"
    "targeted_triple_cw_cwm_owns"
    "targeted_triple_cw_bought_instance"
)

VARIANT="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"

scontrol update \
    JobId="$SLURM_JOB_ID" \
    JobName="Ablation_${VARIANT}" \
    2>/dev/null || true


# ------------------------------------------------------------------------------
# 4. Job information
# ------------------------------------------------------------------------------

echo "================================================================="
echo "  B2B TRANSE ABLATION STUDY"
echo "  Array Task ID : $SLURM_ARRAY_TASK_ID"
echo "  Variant       : $VARIANT"
echo "  Job ID        : $SLURM_JOB_ID"
echo "  Hostname      : $(hostname)"
echo "  Started on    : $(date)"
echo "  Project root  : $PROJECT_ROOT"
echo "================================================================="


# ------------------------------------------------------------------------------
# 5. Environment setup
# ------------------------------------------------------------------------------

echo "Loading modules..."

module purge
module load gnu8/8.3.0
module load python/3.9.10

echo "Activating virtual environment..."

source "${PROJECT_ROOT}/.venv/bin/activate"

echo "Python version:"
python --version


# ------------------------------------------------------------------------------
# 6. Pre-flight checks
# ------------------------------------------------------------------------------

echo "Checking GPU..."
nvidia-smi || echo "nvidia-smi not available"


# ------------------------------------------------------------------------------
# 7. Run ablation for the assigned variant
# ------------------------------------------------------------------------------

cd "$PROJECT_ROOT"

echo "-----------------------------------------------------------------"
echo "Starting TransE ablation training"
echo "Variant: $VARIANT"
echo "-----------------------------------------------------------------"

python ablation_study/run_transe_ablation_batch.py \
    --variant "$VARIANT"


# ------------------------------------------------------------------------------
# 8. Done
# ------------------------------------------------------------------------------

echo "================================================================="
echo "  JOB FINISHED"
echo "  Variant     : $VARIANT"
echo "  Finished on : $(date)"
echo "================================================================="
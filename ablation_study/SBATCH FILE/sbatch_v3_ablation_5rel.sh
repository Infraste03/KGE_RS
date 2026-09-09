#!/bin/bash
# ==============================================================================
# SLURM SUBMISSION SCRIPT — FASHION v3 TRANSE ABLATION, 5-RELATION KG
# ==============================================================================
# Usage:
#   sbatch sbatch_v3_ablation_5rel.sh
#
# No command-line arguments are required.
# The SLURM job array executes all 13 ablation variants.
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. SLURM settings
# ------------------------------------------------------------------------------

#SBATCH --job-name=TransE_v3_Ablation_5rel
#SBATCH --output=Ablation_v3_5rel_%x_%A_%a.log
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
    "loo_no_compatible_with"
    "loo_no_belongs_to"
    "loo_no_belongs_to_brand"
    "loo_no_belongs_to_price_tier"
    "loo_no_belongs_to_pop_tier"
    "mix1_keep_cat_pop"
    "mix2_keep_pop_cw"
    "mix3_keep_cat_brand_cw"
    "targeted_pair_cw_brand"
    "targeted_pair_cw_category"
    "targeted_single_cw"
    "targeted_triple_cw_brand_pricetier"
    "targeted_triple_cw_category_poptier"
)

VARIANT="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"

scontrol update \
    JobId="$SLURM_JOB_ID" \
    JobName="Ablation_v3_5rel_${VARIANT}" \
    2>/dev/null || true


# ------------------------------------------------------------------------------
# 4. Job information
# ------------------------------------------------------------------------------

echo "================================================================="
echo "  FASHION v3 — TRANSE ABLATION — 5-RELATION KG"
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
echo "Starting Fashion v3 TransE ablation"
echo "KG family: 5-rel"
echo "Variant  : $VARIANT"
echo "-----------------------------------------------------------------"

python ablation_study/fashionv3/run_v3_ablation_5rel.py \
    --variant "$VARIANT"


# ------------------------------------------------------------------------------
# 8. Done
# ------------------------------------------------------------------------------

echo "================================================================="
echo "  JOB FINISHED"
echo "  Variant     : $VARIANT"
echo "  Finished on : $(date)"
echo "================================================================="
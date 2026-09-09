#!/bin/bash
# ==============================================================================
# SLURM SUBMISSION SCRIPT FOR STEP 4 ALTERNATE LEARNING — ABLATION V2 (JOB ARRAY)
# ==============================================================================
# Usage:
#   sbatch --array=1-3 sbatch_step4_ablation_v2.sh
# (relaunches ONLY variants that failed due to the vocabulary bug:
#  1=loo_no_instance_of, 2=mix1_keep_compatible_with_owns,
#  3=mix3_keep_bought_compatible_with)
# ==============================================================================

# --- 1. SLURM settings ---

#SBATCH --job-name=Step4_Ablation_v2
#SBATCH --output=Step4_Ablation_v2_%x_%A_%a.log
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=50G
#SBATCH --time=0-23:59:0
#SBATCH --array=0-4

# --- 2. Mappatura indice -> nome variante ---

VARIANTS=(
    "loo_no_compatible_with"
    "loo_no_instance_of"
    "mix1_keep_compatible_with_owns"
    "mix3_keep_bought_compatible_with"
    "targeted_triple_cw_bought_instance"
)

VARIANT="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"

scontrol update JobId="$SLURM_JOB_ID" JobName="Step4AblV2_${VARIANT}" 2>/dev/null || true

# --- 3. Banner ---

echo "================================================================="
echo "  STEP 4 ALTERNATE LEARNING - ABLATION STUDY (V2, fix vocabolario)"
echo "  Array Task ID : $SLURM_ARRAY_TASK_ID"
echo "  Variant       : $VARIANT"
echo "  Scheduling    : one_to_one (fisso, da step4_seeds_one_to_one.yaml)"
echo "  Job ID        : $SLURM_JOB_ID"
echo "  Hostname      : $(hostname)"
echo "  Started on    : $(date)"
echo "  Working dir   : $(pwd)"
echo "================================================================="

# --- 4. Environment setup ---

echo "Loading modules..."
module purge
module load gnu8/8.3.0
module load python/3.9.10

echo "Activating VENV..."
source .venv/bin/activate

echo "Python AFTER activating venv:"
python --version

# --- 5. Pre-flight checks ---

echo "Checking GPU with nvidia-smi..."
nvidia-smi || echo "nvidia-smi not available"

# --- 6. Run Step 4 alternate learning ablation (v2) for the assigned variant ---

echo "-----------------------------------------------------------------"
echo "Starting Step 4 alternate learning (ablation v2) for variant: $VARIANT"
echo "-----------------------------------------------------------------"

python B2B_alternate_learning/run_step4_ablation.py \
    --config B2B_alternate_learning/configs/step4_seeds_one_to_one.yaml \
    --variant "$VARIANT"

# --- 7. Done ---

echo "================================================================="
echo "  JOB FINISHED"
echo "  Variant     : $VARIANT"
echo "  Finished on : $(date)"
echo "================================================================="
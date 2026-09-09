#!/bin/bash
# ============================================================================
# SLURM — STEP 4 FASHION HPO (ALTERNATE LEARNING)
# ============================================================================
# Usage:
#   sbatch ext_one_sbatch_hpo_step4_fashion.sh
# ============================================================================

#SBATCH --job-name=Step4_Fashion_HPO_one_to_one_ext_v1
#SBATCH --output=HPO_ext_v1_one_to_one_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=50G
#SBATCH --time=0-23:59:0

SCHEDULING="one_to_one"

STUDY_NAME="step4_fashion_${SCHEDULING}_ext_v1"
STUDY_DIR="/path/to/KG_RS/fashion_generalization/alternate_learning/hpo_results/${STUDY_NAME}"

echo "================================================================="
echo "  STEP 4 FASHION — HPO (ALTERNATE LEARNING)"
echo "  Scheduling  : $SCHEDULING"
echo "  Study name  : $STUDY_NAME"
echo "  Study dir   : $STUDY_DIR"
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Hostname    : $(hostname)"
echo "  Started on  : $(date)"
echo "  Working dir : $(pwd)"
echo "================================================================="

echo "Loading modules..."
module purge
module load gnu8/8.3.0
module load python/3.9.10

echo "Activating VENV..."
source /path/to/KG_RS/.venv/bin/activate

echo "Python:"
python --version

echo "GPU:"
nvidia-smi || echo "nvidia-smi not available"

echo "-----------------------------------------------------------------"
echo "Starting HPO fashion scheduling=$SCHEDULING"
echo "-----------------------------------------------------------------"

mkdir -p "$STUDY_DIR"

python fashion_generalization/alternate_learning/ext_one_run_hpo_step4_fashion.py \
    --base-config fashion_generalization/alternate_learning/configs/step4_hpc_config_fashion.yaml \
    --scheduling "$SCHEDULING" \
    --study-name "$STUDY_NAME" \
    --study-dir  "$STUDY_DIR" \
    ${HPO_TRIAL_TIMEOUT:+--timeout "$HPO_TRIAL_TIMEOUT"}

echo "================================================================="
echo "  JOB FINISHED"
echo "  Scheduling  : $SCHEDULING"
echo "  Finished on : $(date)"
echo "================================================================="
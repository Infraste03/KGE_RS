#!/bin/bash
#SBATCH --job-name=Step4_Abl_Seed_v1
#SBATCH --output=Step4_Abl_Seed_v1_%x_%A_%a.log
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=50G
#SBATCH --time=0-23:59:0
#SBATCH --array=0-7

VARIANTS=("loo_no_compatible_with" "targeted_triple_cw_bought_instance")
SEEDS=(42 1024 999 2024)

N_SEEDS=${#SEEDS[@]}
VARIANT_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))

VARIANT="${VARIANTS[$VARIANT_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
RUN_TAG="_seed${SEED}"

scontrol update JobId="$SLURM_JOB_ID" JobName="AblV1_${VARIANT}_s${SEED}" 2>/dev/null || true

echo "================================================================="
echo "  STEP 4 ABLATION V1 - MULTI-SEED RUN"
echo "  Array Task ID : $SLURM_ARRAY_TASK_ID"
echo "  Variant       : $VARIANT"
echo "  Seed          : $SEED"
echo "  Job ID        : $SLURM_JOB_ID"
echo "  Hostname      : $(hostname)"
echo "  Started on    : $(date)"
echo "================================================================="

module purge
module load gnu8/8.3.0
module load python/3.9.10
source .venv/bin/activate
nvidia-smi || echo "nvidia-smi not available"

python B2B_alternate_learning/run_step4_ablation.py \
    --config B2B_alternate_learning/configs/step4_seeds_one_to_one.yaml \
    --variant "$VARIANT" \
    --seed "$SEED" \
    --run-tag "$RUN_TAG"

echo "  JOB FINISHED: $VARIANT seed=$SEED  on $(date)"
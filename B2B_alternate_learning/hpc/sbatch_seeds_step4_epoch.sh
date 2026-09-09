#!/bin/bash
#SBATCH --job-name=Step4_seed_epoch
#SBATCH --output=SEED_%x_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=50G
#SBATCH --time=0-23:59:0

# --- 2. Argument parsing ---
SEED="${1:?ERROR: pass the seed as an argument. Es: sbatch sbatch_seeds_step4_epoch.sh 42}"

CONFIG="B2B_alternate_learning/configs/step4_seeds_epoch.yaml"
SCHEDULING="epoch"
RUN_TAG="_seeds_seed${SEED}"

scontrol update JobId="$SLURM_JOB_ID" JobName="Step4_epoch_seed${SEED}" 2>/dev/null || true

echo "================================================================="
echo "  STEP 4 - FINAL 5-SEED RUN (epoch scheduling)"
echo "  Seed        : $SEED"
echo "  Config      : $CONFIG"
echo "  Output dir  : hpc_step4_results/step4_${SCHEDULING}${RUN_TAG}"
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Started on  : $(date)"
echo "================================================================="

module purge
module load gnu8/8.3.0
module load python/3.9.10
source .venv/bin/activate
python --version
nvidia-smi || echo "nvidia-smi not available"

python B2B_alternate_learning/run_step4.py \
    --config "$CONFIG" \
    --scheduling "$SCHEDULING" \
    --seed "$SEED" \
    --run-tag "$RUN_TAG" \
    --epochs 50

echo "================================================================="
echo "  RUN FINISHED | Seed: $SEED | $(date)"
echo "================================================================="
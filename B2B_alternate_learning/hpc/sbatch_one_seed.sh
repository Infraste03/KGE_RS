#!/bin/bash
# ============================================================================
# SLURM SUBMISSION SCRIPT — STEP 4 FINAL 5-SEED RUNS (one_to_one)
# ============================================================================
# Run a full run (50 epochs) with hyperparameter configuration
# best found via HPO, using a seed passed as an argument.
# All checkpoints from each epoch are saved in the seed<SEED> folder.
#
# Usage:
#   sbatch sbatch_seeds_step4.sh <SEED>
#
# Example (launches the 5 seeds):
#   sbatch sbatch_seeds_step4.sh 2020
#   sbatch sbatch_seeds_step4.sh 42
#   sbatch sbatch_seeds_step4.sh 123
#   sbatch sbatch_seeds_step4.sh 999
#   sbatch sbatch_seeds_step4.sh 2024
# ============================================================================

# --- 1. SLURM settings ---
#SBATCH --job-name=Step4_seed
#SBATCH --output=SEED_%x_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=50G
#SBATCH --time=0-23:59:0

# --- 2. Argument parsing ---
SEED="${1:?ERROR: pass the seed as an argument. Es: sbatch sbatch_one_seed.sh 42}"
CONFIG="B2B_alternate_learning/configs/step4_seeds_one_to_one.yaml"
SCHEDULING="one_to_one"
RUN_TAG="_seeds_seed${SEED}"   # directory: hpc_step4_results/step4_one_to_one_seeds_seed<SEED>

# rinomina job in SLURM accounting
scontrol update JobId="$SLURM_JOB_ID" JobName="Step4_seed${SEED}" 2>/dev/null || true

# --- 3. Banner ---
echo "================================================================="
echo "  STEP 4 - FINAL 5-SEED RUN"
echo "  Scheduling  : $SCHEDULING"
echo "  Seed        : $SEED"
echo "  Config      : $CONFIG"
echo "  Run tag     : $RUN_TAG"
echo "  Output dir  : hpc_step4_results/step4_${SCHEDULING}${RUN_TAG}"
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Hostname    : $(hostname)"
echo "  Started on  : $(date)"
echo "  Working dir : $(pwd)"
echo "================================================================="

# --- 4. Environment setup ---
echo "Loading modules..."
module purge
module load gnu8/8.3.0
module load python/3.9.10

echo "Activating VENV..."
source .venv/bin/activate
python --version

# --- 5. Pre-flight checks ---
echo "Checking GPU..."
nvidia-smi || echo "nvidia-smi not available"

# --- 6. Run ---
echo "-----------------------------------------------------------------"
echo "Avvio run con seed=$SEED"
echo "-----------------------------------------------------------------"

python B2B_alternate_learning/run_step4.py \
    --config "$CONFIG" \
    --scheduling "$SCHEDULING" \
    --seed "$SEED" \
    --run-tag "$RUN_TAG" \
    --epochs 50

# --- 7. Done ---
echo "================================================================="
echo "  RUN FINISHED"
echo "  Seed        : $SEED"
echo "  Finished on : $(date)"
echo "================================================================="
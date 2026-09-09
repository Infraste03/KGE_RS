#!/bin/bash
# ==============================================================================
# SLURM SUBMISSION SCRIPT FOR STEP 4 (ALTERNATE LEARNING)
# ==============================================================================
# Usage:
#   sbatch sbatch_step4_alternate.sh one_to_one
#   sbatch sbatch_step4_alternate.sh epoch
#   sbatch sbatch_step4_alternate.sh adaptive
#
# Or, to launch all three in parallel in one go:
#   for s in one_to_one epoch adaptive; do sbatch sbatch_step4_alternate.sh $s; done
# ==============================================================================

# --- 1. Settings for the SLURM Scheduler ---

#SBATCH --job-name=Step4_Alternate
#SBATCH --output=Step4_%x_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=50G
#SBATCH --time=0-23:59:0

# --- 2. Argument parsing ---

SCHEDULING="${1:-one_to_one}"  # default to one_to_one if no argument is given

VALID_SCHEDULING=("one_to_one" "epoch" "adaptive")
SCHEDULING_VALID=0
for s in "${VALID_SCHEDULING[@]}"; do
    if [[ "$SCHEDULING" == "$s" ]]; then
        SCHEDULING_VALID=1
        break
    fi
done

if [[ $SCHEDULING_VALID -eq 0 ]]; then
    echo "ERROR: invalid scheduling '$SCHEDULING'"
    echo "Valid options: ${VALID_SCHEDULING[@]}"
    exit 1
fi

# Override SLURM job name for clearer output filenames
scontrol update JobId=$SLURM_JOB_ID JobName=Step4_${SCHEDULING} 2>/dev/null || true

# --- 3. Banner ---

echo "================================================================="
echo "  STEP 4 - ALTERNATE LEARNING"
echo "  Scheduling  : $SCHEDULING"
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

echo "Python AFTER activating venv:"
python --version

# --- 5. Pre-flight checks ---

echo "Checking GPU with nvidia-smi..."
nvidia-smi || echo "nvidia-smi not available"

# --- 6. Run Alternate Learning for the selected scheduling ---

echo "-----------------------------------------------------------------"
echo "Starting Alternate Learning for scheduling: $SCHEDULING"
echo "-----------------------------------------------------------------"

# Assumi che i file siano correttamente messi nei path in step4_hpc_config.yaml
python B2B_alternate_learning/run_step4.py \
    --config B2B_alternate_learning/configs/step4_hpc_config.yaml \
    --scheduling "$SCHEDULING"

# --- 7. Done ---

echo "================================================================="
echo "  JOB FINISHED"
echo "  Scheduling  : $SCHEDULING"
echo "  Finished on : $(date)"
echo "================================================================="
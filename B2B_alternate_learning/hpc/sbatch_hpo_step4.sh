#!/bin/bash
# ============================================================================
# SLURM SUBMISSION SCRIPT FOR STEP 4 HPO (ALTERNATE LEARNING)
# ============================================================================
# Usage:
#   sbatch sbatch_hpo_step4.sh [scheduling]
#
# Examples:
#   sbatch sbatch_hpo_step4.sh one_to_one
#   sbatch sbatch_hpo_step4.sh epoch
#   sbatch sbatch_hpo_step4.sh adaptive
# ============================================================================

# --- 1. SLURM settings ---

#SBATCH --job-name=Step4_HPO
#SBATCH --output=HPO_%x_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=50G
#SBATCH --time=0-23:59:0

# --- 2. Argument parsing ---

SCHEDULING="${1:-one_to_one}"
STUDY_NAME="step4_${SCHEDULING}"
STUDY_DIR="hpo_results/${STUDY_NAME}"

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

# Make the job name a bit clearer in SLURM accounting/output.
scontrol update JobId="$SLURM_JOB_ID" JobName="Step4_HPO_${SCHEDULING}" 2>/dev/null || true

# Always run from the repository root so the relative paths in the YAML stay valid.
#cd "$(dirname "$0")/.."

# --- 3. Banner ---

echo "================================================================="
echo "  STEP 4 - HPO (ALTERNATE LEARNING)"
echo "  Scheduling  : $SCHEDULING"
echo "  Study name  : $STUDY_NAME"
echo "  Study dir   : $STUDY_DIR"
echo "  HPO defaults: trials / epochs / seeds from run_hpo_step4.py"
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

# --- 6. Run HPO ---

echo "-----------------------------------------------------------------"
echo "Starting HPO for scheduling: $SCHEDULING"
echo "Study name: $STUDY_NAME"
echo "Study dir : $STUDY_DIR"
echo "-----------------------------------------------------------------"

mkdir -p "$STUDY_DIR"

python B2B_alternate_learning/run_hpo_step4.py \
    --scheduling "$SCHEDULING" \
    --study-name "$STUDY_NAME" \
    --study-dir "$STUDY_DIR" \
    ${HPO_TRIAL_TIMEOUT:+--timeout "$HPO_TRIAL_TIMEOUT"}

# --- 7. Done ---

echo "================================================================="
echo "  JOB FINISHED"
echo "  Scheduling  : $SCHEDULING"
echo "  Study name  : $STUDY_NAME"
echo "  Finished on  : $(date)"
echo "================================================================="

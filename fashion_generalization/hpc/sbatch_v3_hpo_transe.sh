#!/bin/bash
# ==============================================================================
# SLURM SUBMISSION SCRIPT — FASHION v3 HPO TRANSE (JOB ARRAY, 2 VARIANTI)
# ==============================================================================
# Usage:
#   sbatch sbatch_v3_hpo_transe.sh
# (nessun argomento: il job array gestisce entrambe le varianti, 3rel e 5rel)
# ==============================================================================

# --- 1. SLURM settings ---

#SBATCH --job-name=TransE_v3_HPO
#SBATCH --output=V3_HPO_%x_%A_%a.log
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=50G
#SBATCH --time=0-23:59:0
#SBATCH --array=0-1

# --- 2. Mappatura indice -> nome variante ---

VARIANTS=(
    "3rel"
    "5rel"
)

VARIANT="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"

scontrol update JobId="$SLURM_JOB_ID" JobName="V3_HPO_${VARIANT}" 2>/dev/null || true

# --- 3. Banner ---

echo "================================================================="
echo "  FASHION v3 — HPO TRANSE"
echo "  Array Task ID : $SLURM_ARRAY_TASK_ID"
echo "  Variant       : $VARIANT"
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

# --- 6. Run HPO for the assigned variant ---

echo "-----------------------------------------------------------------"
echo "Starting Fashion v3 TransE HPO for variant: $VARIANT"
echo "-----------------------------------------------------------------"

python /path/to/KG_RS/fashion_generalization/models/TransE/run_v3_hpo_transe.py --variant "$VARIANT"

# --- 7. Done ---

echo "================================================================="
echo "  JOB FINISHED"
echo "  Variant     : $VARIANT"
echo "  Finished on : $(date)"
echo "================================================================="
#!/bin/bash
# ============================================================================
# SLURM — SASRec Fashion RETRAINING (5 seeds for Wilcoxon)
# ============================================================================
# Usage (1 seed per job):
#   sbatch sbatch_sasrec_fashion_seeds.sh 2020
#   sbatch sbatch_sasrec_fashion_seeds.sh 42
#   sbatch sbatch_sasrec_fashion_seeds.sh 1024
#   sbatch sbatch_sasrec_fashion_seeds.sh 999
#   sbatch sbatch_sasrec_fashion_seeds.sh 2024
# ============================================================================

#SBATCH --job-name=SASRec_Fashion_seeds
#SBATCH --output=SASREC_Fashion_%x_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=50G
#SBATCH --time=0-23:59:0

SEED="${1:?ERROR: provide a seed. Example: sbatch sbatch_sasrec_fashion_seeds.sh 42}"

BASE_DIR="/path/to/KG_RS/fashion_generalization"
CONFIG="${BASE_DIR}/alternate_learning/configs/sasrec_fashion_seeds.yaml"
DATA_PATH="${BASE_DIR}/data/recbole"

echo "================================================================="
echo "  SASRec Fashion RETRAINING"
echo "  Seed        : $SEED"
echo "  Config      : $CONFIG"
echo "  Data path   : $DATA_PATH"
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Started on  : $(date)"
echo "================================================================="

module purge
module load gnu8/8.3.0
module load python/3.9.10
source /path/to/KG_RS/.venv/bin/activate
python --version
nvidia-smi || echo "nvidia-smi not available"

run_seed() {
    local SEED=$1
    local OUTPUT_DIR="${BASE_DIR}/sasrec_fashion_seeds_results/seed${SEED}"
    mkdir -p "$OUTPUT_DIR"

    echo "-----------------------------------------------------------------"
    echo "  Starting seed=$SEED -> $OUTPUT_DIR"
    echo "-----------------------------------------------------------------"

    scontrol update JobId="$SLURM_JOB_ID" \
        JobName="SASRec_Fashion_seed${SEED}" 2>/dev/null || true

    python -c "
import torch
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'weights_only': False})
from recbole.quick_start import run_recbole
run_recbole(
    model='SASRec',
    dataset='fashion',
    config_file_list=['${CONFIG}'],
    config_dict={
        'seed': ${SEED},
        'data_path': '${DATA_PATH}',
        'checkpoint_dir': '${OUTPUT_DIR}',
        'log_wandb': False,
        'show_progress': False,
    }
)
"
    echo "  Seed=$SEED completed: $(date)"
}

run_seed "$SEED"

echo "================================================================="
echo "  JOB FINISHED | $(date)"
echo "================================================================="
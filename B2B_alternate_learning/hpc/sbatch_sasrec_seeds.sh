#!/bin/bash
# ============================================================================
# SLURM SUBMISSION SCRIPT — SASRec 400-dim RETRAINING (5 for Wilcoxon)
# ============================================================================
# Usage:
#   sbatch sbatch_sasrec_seeds.sh <SEED>
#
# Launch the 5 seeds:
#   sbatch sbatch_sasrec_seeds.sh 2020
#   sbatch sbatch_sasrec_seeds.sh 42
#   sbatch sbatch_sasrec_seeds.sh 999
#   sbatch sbatch_sasrec_seeds.sh 1024
#   sbatch sbatch_sasrec_seeds.sh 2024
# ============================================================================

#SBATCH --job-name=SASRec_seed
#SBATCH --output=SASREC_%x_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=50G
#SBATCH --time=0-23:59:0

SEED="${1:?ERROR: pass the seed as an argument. Es: sbatch sbatch_sasrec_seeds.sh 42}"

OUTPUT_DIR="sasrec_seeds_results/seed${SEED}"
CONFIG="B2B_alternate_learning/configs/sasrec400_seeds.yaml"

scontrol update JobId="$SLURM_JOB_ID" JobName="SASRec_seed${SEED}" 2>/dev/null || true

echo "================================================================="
echo "  SASRec 400-dim RETRAINING"
echo "  Seed        : $SEED"
echo "  Output dir  : $OUTPUT_DIR"
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Started on  : $(date)"
echo "================================================================="

module purge
module load gnu8/8.3.0
module load python/3.9.10
source .venv/bin/activate
python --version
nvidia-smi || echo "nvidia-smi not available"

mkdir -p "$OUTPUT_DIR"

python -c "
from recbole.quick_start import run_recbole
run_recbole(
    model='SASRec',
    dataset='b2b_data',
    config_file_list=['$CONFIG'],
    config_dict={
        'seed': $SEED,
        'data_path': 'dataset/',
        'checkpoint_dir': '$OUTPUT_DIR',
        'log_wandb': False,
        'show_progress': False,
    }
)
"

echo "================================================================="
echo "  SASRec FINISHED | Seed: $SEED | $(date)"
echo "================================================================="
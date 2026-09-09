#!/bin/bash
#SBATCH --job-name=fashion_step4_full5rel
#SBATCH --output=/path/to/KG_RS/fashion_generalization/alternate_learning/step4_full5rel_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=150G
#SBATCH --time=0-23:59:0


SCHEDULING=${1:-one_to_one}

echo "================================================================="
echo "  STEP 4 FASHION — ALTERNATE LEARNING KG full 5 reletionship (CE loss)"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Scheduling: $SCHEDULING"
echo "  Started: $(date)"
echo "================================================================="

module purge
module load gnu8/8.3.0
module load python/3.9.10
source /path/to/KG_RS/.venv/bin/activate

echo "Verifica file input..."
ls -lh /path/to/KG_RS/fashion_generalization/data/processed_v2/kg_train.tsv
ls -lh /path/to/KG_RS/fashion_generalization/results/hpo_transe_full5rel_v2/best_model.pt

cd /path/to/KG_RS/fashion_generalization/alternate_learning

python run_step4_full5rel_fashion.py \
    --config configs/step4_hpc_config_fashion_ce.yaml \
    --scheduling $SCHEDULING

echo "================================================================="
echo "  JOB FINISHED: $(date) | Scheduling: $SCHEDULING"
echo "================================================================="
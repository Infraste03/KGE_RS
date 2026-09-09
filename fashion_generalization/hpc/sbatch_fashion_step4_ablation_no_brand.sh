#!/bin/bash
#SBATCH --job-name=fashion_step4_ablation_no_brand
#SBATCH --output=/path/to/KG_RS/fashion_generalization/alternate_learning/step4_ablation_no_brand_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=150G
#SBATCH --time=0-23:59:0

echo "================================================================="
echo "  STEP 4 FASHION — ALTERNATE LEARNING ABLATION: NO_BRAND (CE loss)"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Started: $(date)"
echo "================================================================="

module purge
module load gnu8/8.3.0
module load python/3.9.10
source /path/to/KG_RS/.venv/bin/activate

cd /path/to/KG_RS/fashion_generalization/alternate_learning

python run_step4_ablation_fashion.py \
    --config configs/step4_hpc_config_fashion_ce.yaml \
    --variant no_brand

echo "================================================================="
echo "  JOB FINISHED: $(date)"
echo "================================================================="
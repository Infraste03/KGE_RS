#!/bin/bash
#SBATCH --job-name=KGE_HPO_fashion_no_category
#SBATCH --output=/path/to/KG_RS/fashion_generalization/models/TransE/KGE_HPO_fashion_no_category_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=150G
#SBATCH --time=0-23:59:0

echo "================================================================="
echo "  HPO TransE Fashion — ABLATION: NO CATEGORY (2 relazioni)"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Started: $(date)"
echo "================================================================="

module purge
module load gnu8/8.3.0
module load python/3.9.10
source /path/to/KG_RS/.venv/bin/activate

python -c "import torch, pandas, optuna, pykeen; print('ALL LIBS OK'); print('CUDA:', torch.cuda.is_available())"
nvidia-smi || echo "nvidia-smi not available"

echo "Verifica file input (data/processed/)..."
ls -lh /path/to/KG_RS/fashion_generalization/data/processed/kg_train.tsv
ls -lh /path/to/KG_RS/fashion_generalization/data/processed/taskA_valid.tsv
ls -lh /path/to/KG_RS/fashion_generalization/data/processed/taskA_test.tsv

cd /path/to/KG_RS/fashion_generalization/models/TransE
python run_hpo_transe_fashion_no_category.py

echo "================================================================="
echo "  JOB FINISHED: $(date)"
echo "================================================================="
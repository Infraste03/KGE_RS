#!/bin/bash
#SBATCH --job-name=KGE_HPO_fashion_full
#SBATCH --output=/path/to/KG_RS/fashion_generalization/models/TransE/KGE_HPO_fashion_full_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=150G
#SBATCH --time=0-23:59:0

echo "================================================================="
echo "  HPO TransE Fashion — KG complete (5 relazioni)"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Started: $(date)"
echo "================================================================="

module purge
module load gnu8/8.3.0
module load python/3.9.10
source /path/to/KG_RS/.venv/bin/activate

python -c "import torch, pandas, optuna, pykeen; print('ALL LIBS OK'); print('CUDA:', torch.cuda.is_available())"
nvidia-smi || echo "nvidia-smi not available"

echo "Verify file input (data/processed_v2/)..."
ls -lh /path/to/KG_RS/fashion_generalization/data/processed_v2/kg_train.tsv
ls -lh /path/to/KG_RS/fashion_generalization/data/processed_v2/taskA_valid.tsv
ls -lh /path/to/KG_RS/fashion_generalization/data/processed_v2/taskA_test.tsv

cd /path/to/KG_RS/fashion_generalization/models/TransE
python run_full_hpo_transe_fashion.py

echo "================================================================="
echo "  JOB FINISHED: $(date)"
echo "================================================================="
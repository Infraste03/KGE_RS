#!/bin/bash
# ==============================================================================
# SLURM SUBMISSION SCRIPT FOR KGE HPO — FASHION GENERALIZATION
# ==============================================================================

#SBATCH --job-name=KGE_HPO_fashion
#SBATCH --output=KGE_HPO_fashion_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=150G
#SBATCH --time=0-23:59:0

echo "================================================================="
echo "  HYPERPARAMETER OPTIMIZATION (HPO) JOB STARTED — FASHION"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Executed on: $(hostname)"
echo "  Started on: $(date)"
echo "  Working directory: $(pwd)"
echo "================================================================="

echo "Loading modules..."
module purge
module load gnu8/8.3.0
module load python/3.9.10

echo "Python BEFORE activating venv:"
which python
python --version

echo "Activating VENV..."
source /path/to/KG_RS/.venv/bin/activate

echo "Python AFTER activating venv:"
which python
python --version
pip --version

echo "Checking libraries..."
python -c "import torch, pandas, numpy, optuna, pykeen; print('ALL LIBS OK'); print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo "Checking GPU with nvidia-smi..."
nvidia-smi || echo "nvidia-smi not available"

echo "Checking input files..."
ls -lh /path/to/KG_RS/fashion_generalization/data/processed/kg_train.tsv
ls -lh /path/to/KG_RS/fashion_generalization/data/processed/taskA_valid.tsv
ls -lh /path/to/KG_RS/fashion_generalization/data/processed/taskA_test.tsv

echo "-----------------------------------------------------------------"
echo "Starting Hyperparameter Optimization (HPO) for TransE — Fashion..."

cd /path/to/KG_RS/fashion_generalization/models/TransE
python run_hpo_transe_fashion.py

echo "================================================================="
echo "  JOB FINISHED"
echo "  Finished on: $(date)"
echo "================================================================="
#!/bin/bash
# ==============================================================================
# SLURM SUBMISSION SCRIPT FOR KGE-TRAIN HYPERPARAMETER OPTIMIZATION (HPO)
# ==============================================================================

#SBATCH --job-name=KGE-TRAIN_HPO
#SBATCH --output=KGE-TRAIN_HPO_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=150G
#SBATCH --time=0-23:59:0

echo "================================================================="
echo "  HYPERPARAMETER OPTIMIZATION (HPO) JOB STARTED"
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
source .venv/bin/activate

echo "Python AFTER activating venv:"
which python
python --version
pip --version

echo "Checking libraries..."
python -c "import torch, pandas, numpy, optuna, pykeen; print('ALL LIBS OK'); print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo "Checking GPU with nvidia-smi..."
nvidia-smi || echo "nvidia-smi not available"

echo "Checking input files..."
ls -lh data/processed/kg_train.tsv
ls -lh data/processed/taskA_valid.tsv
ls -lh data/processed/taskA_test.tsv

echo "-----------------------------------------------------------------"
echo "Starting Hyperparameter Optimization (HPO) for KGE-TRAIN..."

python run_hpo_kge.py

echo "================================================================="
echo "  JOB FINISHED"
echo "  Finished on: $(date)"
echo "================================================================="
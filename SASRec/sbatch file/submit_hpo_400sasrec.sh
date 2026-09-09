#!/bin/bash
# ==============================================================================
# SBATCH FILE FOR HYPERPARAMETER OPTIMIZATION (HPO) OF SASRec MODEL
# ==============================================================================

# --- 1. Settings SLURM ---

#SBATCH --job-name=SASRec-HPO   # name  job (HPO per SASRec)
#SBATCH --output=SASRec-HPO_%j.log # name file output
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=150G
#SBATCH --time=0-23:59:0

# --- 2. Setup env ---
echo "================================================================="
echo "  hpo SASRec job started"
echo "  Job ID: $SLURM_JOB_ID"
echo "  running on: $(hostname)"
echo "  started at: $(date)"
echo "  Working directory: $(pwd)"
echo "================================================================="

# --- 3. Environment setup ---

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

# --- 4. Execution of the Python Script ---

echo "Starting hyperparameter optimization for SASRec..."
python run_hpo_sasrec400HPC.py

echo "================================================================="
echo "  JOB COMPLETED"
echo "   Completed at: $(date)"
echo "================================================================="
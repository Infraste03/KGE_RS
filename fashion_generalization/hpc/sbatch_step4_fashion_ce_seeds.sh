#!/bin/bash
# ==============================================================================
# SLURM — Step 4 Fashion v1 CE — multi-seed (entrambi gli scheduling)
# Iperparametri FISSI (identici a trial 0 HPO CE, gia' nel YAML default):
#   lr=0.0001, batch_size_kge=1024, batch_size_sasrec=256, margin_loss=8.684,
#   hidden_dropout=0.5, attn_dropout=0.3
# Benchmark seed=2020: one_to_one R@20=0.1039/NDCG@20=0.0908,
#                       epoch      R@20=0.1039/NDCG@20=0.0914
# ==============================================================================
# Uso:
#   sbatch sbatch_step4_fashion_ce_seeds.sh <scheduling> <seed>
#   sbatch sbatch_step4_fashion_ce_seeds.sh one_to_one 42
#   sbatch sbatch_step4_fashion_ce_seeds.sh epoch 42
# ==============================================================================

#SBATCH --job-name=step4_fashion_ce_seeds
#SBATCH --output=/path/to/KG_RS/fashion_generalization/alternate_learning/hpc_step4_results/step4_ce_seeds_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=150G
#SBATCH --time=0-23:59:0

SCHEDULING=${1:?"ERRORE: specifica scheduling (one_to_one o epoch). Uso: sbatch sbatch_step4_fashion_ce_seeds.sh <scheduling> <seed>"}
SEED=${2:?"ERRORE: specifica seed. Uso: sbatch sbatch_step4_fashion_ce_seeds.sh <scheduling> <seed>"}

if [[ "${SCHEDULING}" != "one_to_one" && "${SCHEDULING}" != "epoch" ]]; then
    echo "ERRORE: scheduling deve essere one_to_one o epoch, ricevuto: ${SCHEDULING}"
    exit 1
fi

echo "================================================================="
echo "  STEP 4 FASHION v1 CE — seed multipli"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Scheduling: ${SCHEDULING}"
echo "  Seed: ${SEED}"
echo "  Started on: $(date)"
echo "================================================================="

module purge
module load gnu8/8.3.0
module load python/3.9.10
source /path/to/KG_RS/.venv/bin/activate
echo "Python:"; python --version
nvidia-smi || echo "nvidia-smi not available"

BASE=/path/to/KG_RS
FASHION_BASE=${BASE}/fashion_generalization
ALT_DIR=${FASHION_BASE}/alternate_learning

echo "-----------------------------------------------------------------"
echo "Checking input files..."
ls -lh ${ALT_DIR}/configs/step4_config_fashion.yaml
ls -lh ${FASHION_BASE}/results/hpo_transe/best_model.pt
ls -lh ${FASHION_BASE}/results/sasrec_hpo/trial_017/best_valid.pth

echo "-----------------------------------------------------------------"
echo "Starting Step 4 Fashion v1 CE (${SCHEDULING}, seed=${SEED})..."

cd ${ALT_DIR}

python run_step4_fashion_ce.py \
    --config configs/step4_config_fashion.yaml \
    --scheduling ${SCHEDULING} \
    --epochs 50 \
    --seed ${SEED} \
    --run-tag "_seed${SEED}"

echo "-----------------------------------------------------------------"
BEST_METRICS="${ALT_DIR}/hpc_step4_results/step4_${SCHEDULING}_ce_seed${SEED}/best_metrics.json"
if [ -f "${BEST_METRICS}" ]; then
    echo "  Run completata. Risultati:"
    cat "${BEST_METRICS}"
else
    echo "  ATTENZIONE: best_metrics.json non trovato."
fi

echo "================================================================="
echo "  JOB FINISHED | $(date)"
echo "================================================================="
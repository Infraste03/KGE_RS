#!/bin/bash
# ==============================================================================
# SLURM — Step 4 Alternate Learning — Fashion v3
# Uso:
#   sbatch sbatch_step4_fashion_v3.sh <relations> [scheduling] [run_tag]
#   sbatch sbatch_step4_fashion_v3.sh 3
#   sbatch sbatch_step4_fashion_v3.sh 5 epoch
#   sbatch sbatch_step4_fashion_v3.sh 3 one_to_one seed42
# ==============================================================================

#SBATCH --job-name=Step4_fashion_v3
#SBATCH --output=/path/to/KG_RS/fashion_generalization/alternate_learning/hpc_step4_results_v3/step4_v3_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=150G
#SBATCH --time=0-23:59:0

RELATIONS=${1:?"Errore: specifica --relations 3 o 5 come primo argomento"}
SCHEDULING=${2:-one_to_one}
RUN_TAG=${3:-}

echo "================================================================="
echo "  STEP 4 — Alternate Learning — Fashion v3"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Executed on: $(hostname)"
echo "  Started on: $(date)"
echo "  Relations: ${RELATIONS}"
echo "  Scheduling: ${SCHEDULING}"
echo "  Run tag: ${RUN_TAG:-<none>}"
echo "================================================================="

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

echo "Checking libraries..."
python -c "import torch, recbole, pykeen, pandas, numpy; print('ALL LIBS OK'); print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo "Checking GPU with nvidia-smi..."
nvidia-smi || echo "nvidia-smi not available"

BASE=/path/to/KG_RS
FASHION_BASE=${BASE}/fashion_generalization
ALT_DIR=${FASHION_BASE}/alternate_learning

echo "-----------------------------------------------------------------"
echo "Checking input files..."
ls -lh ${ALT_DIR}/configs/step4_config_fashion_v3.yaml
ls -lh ${FASHION_BASE}/data/processed_v3/kg_train.tsv
ls -lh ${FASHION_BASE}/data/processed_v3/entity2id.tsv
ls -lh ${FASHION_BASE}/data/processed_v3/relation2id.tsv
ls -lh ${FASHION_BASE}/data/recbole_v3/fashion_v3/fashion_v3.inter
ls -lh ${FASHION_BASE}/models/TransE/${RELATIONS}rel/best_model.pt
ls -lh ${FASHION_BASE}/models/TransE/${RELATIONS}rel/best_params.json

echo "-----------------------------------------------------------------"
echo "Checking output directory doesn't pre-exist (unless resuming)..."
OUT_DIR=${ALT_DIR}/hpc_step4_results_v3/step4_${RELATIONS}rel_${SCHEDULING}_ce${RUN_TAG}
if [ -f "${OUT_DIR}/best_metrics.json" ]; then
    echo "  ATTENZIONE: ${OUT_DIR}/best_metrics.json esiste gia'."
    echo "  Lo script terminera' subito a meno di passare --force manualmente."
elif [ -f "${OUT_DIR}/checkpoint_latest.pth" ]; then
    echo "  Checkpoint di resume trovato: ${OUT_DIR}/checkpoint_latest.pth"
    echo "  Il training riprendera' da li'."
else
    echo "  Nessun output precedente trovato, run pulita."
fi

echo "-----------------------------------------------------------------"
echo "Starting Step 4 Alternate Learning (relations=${RELATIONS}, scheduling=${SCHEDULING})..."

cd ${ALT_DIR}

RUN_TAG_ARG=""
if [ -n "${RUN_TAG}" ]; then
    RUN_TAG_ARG="--run-tag ${RUN_TAG}"
fi

python run_step4_fashion_v3.py \
    --config configs/step4_config_fashion_v3.yaml \
    --relations ${RELATIONS} \
    --scheduling ${SCHEDULING} \
    ${RUN_TAG_ARG}

echo "================================================================="
echo "  JOB FINISHED"
echo "  Finished on: $(date)"
echo "================================================================="
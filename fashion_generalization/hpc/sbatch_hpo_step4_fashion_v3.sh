#!/bin/bash
#SBATCH --job-name=HPO_step4_fashion_v3
#SBATCH --output=/path/to/KG_RS/fashion_generalization/alternate_learning/hpo_results_fashion_v3/hpo_step4_v3_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=150G
#SBATCH --time=0-23:59:0

TRIALS=${1:-30}
EPOCHS=${2:-20}
SCHEDULING=${3:-one_to_one}

echo "================================================================="
echo "  HPO STEP 4 — Alternate Learning — Fashion v3 (3rel)"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Executed on: $(hostname)"
echo "  Started on: $(date)"
echo "  Trials totali: ${TRIALS}"
echo "  Epoche per trial: ${EPOCHS}"
echo "  Scheduling: ${SCHEDULING}"
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
python -c "import torch, recbole, pykeen, optuna, pandas, numpy; print('ALL LIBS OK'); print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo "Checking GPU with nvidia-smi..."
nvidia-smi || echo "nvidia-smi not available"

BASE=/path/to/KG_RS
FASHION_BASE=${BASE}/fashion_generalization
ALT_DIR=${FASHION_BASE}/alternate_learning

echo "-----------------------------------------------------------------"
echo "Checking input files..."
ls -lh ${ALT_DIR}/configs/step4_config_fashion_v3.yaml
ls -lh ${FASHION_BASE}/models/TransE/3rel/best_model.pt
ls -lh ${FASHION_BASE}/models/TransE/3rel/best_params.json

echo "-----------------------------------------------------------------"
echo "Checking sasrec_checkpoint is set in config (not null)..."
python -c "
import yaml
with open('${ALT_DIR}/configs/step4_config_fashion_v3.yaml') as f:
    cfg = yaml.safe_load(f)
ckpt = cfg['paths'].get('sasrec_checkpoint')
if ckpt is None:
    raise SystemExit('ERROR: paths.sasrec_checkpoint and still null in the YAML. Compile it before launching.')
print(f'sasrec_checkpoint = {ckpt}')
"
if [ $? -ne 0 ]; then
    echo "ABORT: config non pronto."
    exit 1
fi

echo "-----------------------------------------------------------------"
echo "Checking resume state..."
STUDY_DIR_GLOB="${ALT_DIR}/hpo_results_fashion_v3"
if [ -d "${STUDY_DIR_GLOB}" ]; then
    LATEST_CSV=$(find "${STUDY_DIR_GLOB}" -name "trials_summary.csv" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    if [ -n "${LATEST_CSV}" ]; then
        N_DONE=$(($(wc -l < "${LATEST_CSV}") - 1))
        echo "  Trial gia' completati (trovati in ${LATEST_CSV}): ${N_DONE}"
        if [ "${N_DONE}" -ge "${TRIALS}" ]; then
            echo "  HPO gia' completo (${N_DONE}/${TRIALS}). Nulla da fare."
            echo "  Se vuoi comunque rilanciare, cancella o rinomina la cartella study."
            exit 0
        fi
    else
        echo "  Nessuno studio precedente trovato, run pulita."
    fi
fi

echo "-----------------------------------------------------------------"
echo "Starting HPO Step 4 Fashion v3..."

cd ${ALT_DIR}

python run_hpo_step4_fashion_v3.py \
    --base-config configs/step4_config_fashion_v3.yaml \
    --trials ${TRIALS} \
    --epochs ${EPOCHS} \
    --scheduling ${SCHEDULING} \
    --storage sqlite:///${ALT_DIR}/hpo_results_fashion_v3/optuna_fashion_v3.db \
    --study-name step4_fashion_v3_${SCHEDULING}_ce

echo "-----------------------------------------------------------------"
echo "Checking final trial count..."
LATEST_CSV=$(find "${STUDY_DIR_GLOB}" -name "trials_summary.csv" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
if [ -n "${LATEST_CSV}" ]; then
    N_DONE=$(($(wc -l < "${LATEST_CSV}") - 1))
    echo "  Trial completati finora: ${N_DONE}/${TRIALS}"
    if [ "${N_DONE}" -lt "${TRIALS}" ]; then
        echo "  ATTENZIONE: HPO non ancora completo (probabile timeout SLURM)."
        echo "  Rilancia lo stesso comando sbatch per continuare il resume."
    fi
fi

echo "================================================================="
echo "  JOB FINISHED"
echo "  Finished on: $(date)"
echo "================================================================="
#!/bin/bash
# ==============================================================================
# SLURM — Step 4 Alternate Learning — Fashion v3 — ABLATION
# Uso:
#   sbatch sbatch_step4_ablation_fashion_v3.sh <variant> [scheduling] [run_tag]
#   sbatch sbatch_step4_ablation_fashion_v3.sh loo_no_compatible_with
#   sbatch sbatch_step4_ablation_fashion_v3.sh mix3_keep_cat_brand_cw epoch
# ==============================================================================

#SBATCH --job-name=Step4_ablation_fashion_v3
#SBATCH --output=/path/to/KG_RS/fashion_generalization/alternate_learning/hpc_step4_results_v3/step4_ablation_v3_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=150G
#SBATCH --time=0-23:59:0

VARIANT=${1:?"Errore: specifica il nome della variante come primo argomento (es. loo_no_compatible_with)"}
SCHEDULING=${2:-one_to_one}
RUN_TAG=${3:-}

VALID_VARIANTS="loo_no_compatible_with mix3_keep_cat_brand_cw targeted_pair_cw_brand loo_no_belongs_to"
if [[ ! " ${VALID_VARIANTS} " =~ " ${VARIANT} " ]]; then
    echo "ERRORE: variante '${VARIANT}' non riconosciuta."
    echo "Valide: ${VALID_VARIANTS}"
    exit 1
fi

echo "================================================================="
echo "  STEP 4 — Alternate Learning — Fashion v3 — ABLATION"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Executed on: $(hostname)"
echo "  Started on: $(date)"
echo "  Variant: ${VARIANT}"
echo "  Scheduling: ${SCHEDULING}"
echo "================================================================="

module purge
module load gnu8/8.3.0
module load python/3.9.10

echo "Activating VENV..."
source /path/to/KG_RS/.venv/bin/activate

echo "Checking libraries..."
python -c "import torch, recbole, pykeen, pandas, numpy; print('ALL LIBS OK'); print('CUDA available:', torch.cuda.is_available())"

echo "Checking GPU with nvidia-smi..."
nvidia-smi || echo "nvidia-smi not available"

BASE=/path/to/KG_RS
FASHION_BASE=${BASE}/fashion_generalization
ALT_DIR=${FASHION_BASE}/alternate_learning
ABLATION_BASE=${BASE}/ablation_study/fashionv3

echo "-----------------------------------------------------------------"
echo "Checking input files for variant '${VARIANT}'..."
ls -lh ${ALT_DIR}/configs/step4_config_fashion_v3.yaml

# Determina la famiglia (3 o 5 relazioni) in base alla variante
case "${VARIANT}" in
    loo_no_belongs_to)
        FAMILY=3
        ;;
    *)
        FAMILY=5
        ;;
esac
echo "  Famiglia: ${FAMILY}rel"

ls -lh ${ABLATION_BASE}/ablation_${FAMILY}rel/kg_train_${VARIANT}.tsv
ls -lh ${ABLATION_BASE}/results_${FAMILY}rel/${VARIANT}/best_model_${VARIANT}.pt

echo "-----------------------------------------------------------------"
echo "Checking sasrec_checkpoint is set in config..."
python -c "
import yaml
with open('${ALT_DIR}/configs/step4_config_fashion_v3.yaml') as f:
    cfg = yaml.safe_load(f)
ckpt = cfg['paths'].get('sasrec_checkpoint')
if ckpt is None:
    raise SystemExit('ERRORE: paths.sasrec_checkpoint e ancora null.')
print(f'sasrec_checkpoint = {ckpt}')
"
if [ $? -ne 0 ]; then
    echo "ABORT: config non pronto."
    exit 1
fi

echo "-----------------------------------------------------------------"
echo "Checking output directory..."
OUT_DIR=${ALT_DIR}/hpc_step4_results_v3/step4_ablation_${VARIANT}_${SCHEDULING}_ce${RUN_TAG}
if [ -f "${OUT_DIR}/best_metrics.json" ]; then
    echo "  ATTENZIONE: ${OUT_DIR}/best_metrics.json esiste gia'."
    echo "  Lo script terminera' subito a meno di passare --force manualmente."
elif [ -f "${OUT_DIR}/checkpoint_latest.pth" ]; then
    echo "  Checkpoint di resume trovato, il training riprendera' da li'."
else
    echo "  Nessun output precedente, run pulita."
fi

echo "-----------------------------------------------------------------"
echo "Starting Step 4 Ablation (variant=${VARIANT}, scheduling=${SCHEDULING})..."

cd ${ALT_DIR}

RUN_TAG_ARG=""
if [ -n "${RUN_TAG}" ]; then
    RUN_TAG_ARG="--run-tag ${RUN_TAG}"
fi

python run_step4_ablation_fashion_v3.py \
    --config configs/step4_config_fashion_v3.yaml \
    --variant ${VARIANT} \
    --scheduling ${SCHEDULING} \
    ${RUN_TAG_ARG}

echo "================================================================="
echo "  JOB FINISHED"
echo "  Finished on: $(date)"
echo "================================================================="
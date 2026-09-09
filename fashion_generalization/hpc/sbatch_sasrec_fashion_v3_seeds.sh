#!/bin/bash
# ============================================================================
# SLURM — SASRec Fashion v3 STANDALONE — multi-seed (per Wilcoxon vs joint model)
# Iperparametri FISSI dal trial 3 vincente HPO SASRec v3
#   (fashion_generalization/results/sasrec_hpo_v3/hpo_results_sasrec_fashion_v3.csv):
#   n_layers=2, n_heads=8, learning_rate=0.00031622776601660001,
#   hidden_dropout_prob=0.3, attn_dropout_prob=0.1, weight_decay=1e-05,
#   loss_type=CE, neg_candidate_num=1
# ============================================================================
# Uso (uno o piu' seed nello stesso job, eseguiti in sequenza):
#   sbatch sbatch_sasrec_fashion_v3_seeds.sh 42
#   sbatch sbatch_sasrec_fashion_v3_seeds.sh 42 1024 999 2024
# ============================================================================

#SBATCH --job-name=SASRec_Fashion_v3_seeds
#SBATCH --output=/path/to/KG_RS/fashion_generalization/sasrec_fashion_v3_seeds_results/sasrec_v3_seeds_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=50G
#SBATCH --time=0-23:59:0

if [ "$#" -eq 0 ]; then
    echo "ERRORE: passa almeno un seed. Es: sbatch sbatch_sasrec_fashion_v3_seeds.sh 42 1024 999 2024"
    exit 1
fi
SEEDS=("$@")

BASE_DIR="/path/to/KG_RS/fashion_generalization"
DATA_PATH="${BASE_DIR}/data/recbole_v3"
RESULTS_BASE="${BASE_DIR}/sasrec_fashion_v3_seeds_results"

echo "================================================================="
echo "  SASRec Fashion v3 STANDALONE — multi-seed"
echo "  Seeds       : ${SEEDS[*]}"
echo "  Data path   : $DATA_PATH"
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Started on  : $(date)"
echo "================================================================="

module purge
module load gnu8/8.3.0
module load python/3.9.10
source /path/to/KG_RS/.venv/bin/activate
echo "Python:"; python --version
echo "Checking GPU..."
nvidia-smi || echo "nvidia-smi not available"

echo "-----------------------------------------------------------------"
echo "Checking input paths..."
ls -lh ${DATA_PATH}/fashion_v3/fashion_v3.inter

run_seed() {
    local SEED=$1
    local OUTPUT_DIR="${RESULTS_BASE}/seed${SEED}"

    if [ -f "${OUTPUT_DIR}/test_result.json" ]; then
        echo "-----------------------------------------------------------------"
        echo "  [SKIP] seed=$SEED gia' completato: ${OUTPUT_DIR}/test_result.json"
        return
    fi

    mkdir -p "$OUTPUT_DIR"

    echo "-----------------------------------------------------------------"
    echo "  Avvio seed=$SEED -> $OUTPUT_DIR"
    echo "-----------------------------------------------------------------"

    scontrol update JobId="$SLURM_JOB_ID" \
        JobName="SASRec_v3_seed${SEED}" 2>/dev/null || true

    python -c "
import json, torch
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'weights_only': False})
from recbole.quick_start import run_recbole

result = run_recbole(
    model='SASRec',
    dataset='fashion_v3',
    config_dict={
        'seed': ${SEED},
        'reproducibility': True,
        'data_path': '${DATA_PATH}',
        'checkpoint_dir': '${OUTPUT_DIR}',
        'show_progress': False,
        'use_gpu': True,
        'log_wandb': False,

        # --- schema campi (identico a step4_config_fashion_v3.yaml) ---
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'TIME_FIELD': 'timestamp',
        'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
        'field_separator': '\t',
        'eval_args': {
            'split': {'LS': 'valid_and_test'},
            'order': 'TO',
            'group_by': 'user',
            'mode': {'valid': 'full', 'test': 'full'},
        },
        'MAX_ITEM_LIST_LENGTH': 50,
        'metrics': ['Recall', 'NDCG'],
        'topk': [20],
        'valid_metric': 'NDCG@20',
        'loss_type': 'CE',
        'train_neg_sample_args': None,

        # --- iperparametri FISSI, trial 3 vincente HPO SASRec v3 ---
        'n_layers': 2,
        'n_heads': 8,
        'hidden_size': 128,
        'embedding_size': 128,
        'inner_size': 256,
        'hidden_dropout_prob': 0.3,
        'attn_dropout_prob': 0.1,
        'learning_rate': 0.00031622776601660001,
        'weight_decay': 1e-05,
    }
)

test_result = result['test_result']
print('TEST RESULT:', test_result)
with open('${OUTPUT_DIR}/test_result.json', 'w') as f:
    json.dump(test_result, f, indent=4)
"
    echo "  Seed=$SEED completato: $(date)"
}

for S in "${SEEDS[@]}"; do
    run_seed "$S"
done

echo "-----------------------------------------------------------------"
echo "Riepilogo finale (tutti i seed passati a questo job):"
for S in "${SEEDS[@]}"; do
    RF="${RESULTS_BASE}/seed${S}/test_result.json"
    if [ -f "$RF" ]; then
        echo "  seed=$S:"
        cat "$RF"
    else
        echo "  seed=$S: MANCANTE (job forse interrotto)"
    fi
done

echo "================================================================="
echo "  JOB FINISHED | $(date)"
echo "================================================================="
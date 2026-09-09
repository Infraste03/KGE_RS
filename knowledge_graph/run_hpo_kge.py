"""
Step 2 (HPO version): Hyperparameter Optimization for KGE models.

Uses PyKEEN with Optuna to search over the most impactful hyperparameters
for TransE, RotatE, ComplEx, and DistMult.

Designed to run on HPC. Each model gets its own Optuna study, saved to
disk so the search can be resumed after preemption.

Optimization objective : MRR (filtered, naive-mode), on validation set
Search algorithm       : Tree-structured Parzen Estimator (TPE)
Test-set protection    : test set is NEVER seen during HPO.
                         Final evaluation on test happens once, after best
                         hyperparameters are selected.

Outputs (under results/step2_hpo/<model_name>/):
    study.db          -- Optuna study database (SQLite, resumable)
    best_params.json  -- best hyperparameter combination
    best_metrics.json -- final test metrics with best hyperparameters
                         (in BOTH naive and type_constrained modes)
    trials.csv        -- summary of all trials for inspection
    run.log           -- per-model training log file
"""

import os
import json
import logging
import time
import argparse
import torch
import numpy as np
import pandas as pd

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.evaluation import RankBasedEvaluator

# ---- root logger setup (file handler is added per-model later) -------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

KG_PATH = os.path.join("data", "processed", "kg_train.tsv")
TASKA_VALID = os.path.join("data", "processed", "taskA_valid.tsv")
TASKA_TEST = os.path.join("data", "processed", "taskA_test.tsv")
RESULTS_DIR = os.path.join("results", "step2_hpo")

MODELS_TO_OPTIMIZE = ['TransE', 'RotatE', 'ComplEx', 'DistMult']

# HPO budget
N_TRIALS_PER_MODEL = 50
TRAINING_EPOCHS_HPO = 50
TRAINING_EPOCHS_FINAL = 300  # upper bound; early stopping will likely cut earlier
RANDOM_SEED = 42

# Time budget per model in seconds (used as Optuna's timeout).
# Leaves margin for the final retraining + evaluation phase.
HPO_TIMEOUT_PER_MODEL = 20 * 3600  # 5 hours

HITS_AT_K = [1, 3, 10]
RELATION_LABEL = 'compatible_with'

# Models that benefit from L2 regularization. TransE and RotatE have built-in
# norm constraints and adding external L2 can destabilize training. ComplEx
# and DistMult are bilinear and benefit from explicit regularization.
MODELS_WITH_L2 = {'ComplEx', 'DistMult'}


# --------------------------------------------------------------------------- #
# Per-model file logging
# --------------------------------------------------------------------------- #

_active_file_handler = None

def attach_file_logger(log_path):
    """Attach a FileHandler so logs are also written to a per-model file.
    Removes any previously attached file handler first."""
    global _active_file_handler
    root = logging.getLogger()
    if _active_file_handler is not None:
        root.removeHandler(_active_file_handler)
        _active_file_handler.close()
    _active_file_handler = logging.FileHandler(log_path, mode='a')
    _active_file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                          datefmt='%Y-%m-%d %H:%M:%S')
    )
    root.addHandler(_active_file_handler)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_triples_factories():
    logger.info("Loading triples factories...")
    train_df = pd.read_csv(KG_PATH, sep='\t')
    valid_df = pd.read_csv(TASKA_VALID, sep='\t')
    test_df = pd.read_csv(TASKA_TEST, sep='\t')

    train_arr = train_df[['head', 'relation', 'tail']].values
    valid_arr = valid_df[['head', 'relation', 'tail']].values
    test_arr = test_df[['head', 'relation', 'tail']].values

    train_factory = TriplesFactory.from_labeled_triples(train_arr)
    valid_factory = TriplesFactory.from_labeled_triples(
        valid_arr,
        entity_to_id=train_factory.entity_to_id,
        relation_to_id=train_factory.relation_to_id,
    )
    test_factory = TriplesFactory.from_labeled_triples(
        test_arr,
        entity_to_id=train_factory.entity_to_id,
        relation_to_id=train_factory.relation_to_id,
    )

    logger.info(f"  Entities : {train_factory.num_entities:,}")
    logger.info(f"  Relations: {train_factory.num_relations:,}")
    logger.info(f"  Triples  : train={train_factory.num_triples:,}, "
                f"valid={valid_factory.num_triples:,}, "
                f"test={test_factory.num_triples:,}")
    return train_factory, valid_factory, test_factory


def build_type_index(train_factory):
    type_to_ids = {'item': [], 'machine': [], 'client': [], 'model': []}
    for label, eid in train_factory.entity_to_id.items():
        prefix = label.split('_')[0]
        if prefix in type_to_ids:
            type_to_ids[prefix].append(eid)
    return {k: torch.LongTensor(v) for k, v in type_to_ids.items()}


# --------------------------------------------------------------------------- #
# Search space
# --------------------------------------------------------------------------- #
#
# Tunable hyperparameters per model:
#   - embedding_dim      (most impactful)
#   - learning_rate      (standard)
#   - num_negs_per_pos   (sample efficiency)
#   - loss_margin        (margin ranking margin)
#   - regularizer_weight (only for ComplEx and DistMult)
#
# Loss is fixed to MarginRankingLoss for all models. This choice is
# conservative but ensures compatibility with the basic negative sampler
# (sLCWA training loop) for all four model architectures, avoiding
# cryptic PyKEEN errors that arise from incompatible (loss, sampler)
# combinations.
# --------------------------------------------------------------------------- #

def suggest_hyperparameters(trial, model_name):
    params = {
        'embedding_dim': trial.suggest_categorical(
            'embedding_dim', [64, 128, 200, 256, 400]
        ),
        'learning_rate': trial.suggest_float(
            'learning_rate', 1e-4, 1e-2, log=True
        ),
        'num_negs_per_pos': trial.suggest_categorical(
            'num_negs_per_pos', [8, 16, 32, 64, 128]
        ),
        'loss_margin': trial.suggest_float('loss_margin', 0.5, 10.0),
    }
    if model_name in MODELS_WITH_L2:
        params['regularizer_weight'] = trial.suggest_float(
            'regularizer_weight', 1e-8, 1e-2, log=True
        )
    return params


def build_pipeline_kwargs(model_name, params, num_epochs, device,
                          use_early_stopping=True):
    """Build kwargs for pykeen.pipeline. Centralizes model-specific logic
    (regularization on/off) so train and retrain stay consistent."""
    kwargs = dict(
        model=model_name,
        model_kwargs={'embedding_dim': params['embedding_dim']},
        loss='marginranking',
        loss_kwargs={'margin': params['loss_margin']},
        optimizer='Adam',
        optimizer_kwargs={'lr': params['learning_rate']},
        training_kwargs={
            'num_epochs': num_epochs,
            'batch_size': 1024,
        },
        negative_sampler='basic',
        negative_sampler_kwargs={'num_negs_per_pos': params['num_negs_per_pos']},
        random_seed=RANDOM_SEED,
        device=device,
        use_tqdm=False,
    )

    if model_name in MODELS_WITH_L2 and 'regularizer_weight' in params:
        kwargs['regularizer'] = 'LpRegularizer'
        kwargs['regularizer_kwargs'] = {
            'weight': params['regularizer_weight'],
            'p': 2,
        }

    if use_early_stopping:
        kwargs['stopper'] = 'early'
        kwargs['stopper_kwargs'] = {
            'frequency': 5,
            'patience': 3,
            'metric': 'inverse_harmonic_mean_rank',
            'relative_delta': 0.002,
        }

    return kwargs


# --------------------------------------------------------------------------- #
# Optuna objective function
# --------------------------------------------------------------------------- #

def make_objective(model_name, train_factory, valid_factory, device):
    def objective(trial):
        params = suggest_hyperparameters(trial, model_name)
        t0 = time.time()

        try:
            kwargs = build_pipeline_kwargs(
                model_name, params, TRAINING_EPOCHS_HPO, device,
                use_early_stopping=True,
            )
            result = pipeline(
                training=train_factory,
                validation=valid_factory,
                testing=valid_factory,  # never use test here -- protected
                evaluator=RankBasedEvaluator(filtered=True),
                **kwargs,
            )

            elapsed = time.time() - t0
            mrr = result.metric_results.to_dict()['both']['realistic'][
                'inverse_harmonic_mean_rank'
            ]

            logger.info(
                f"  [{model_name} trial {trial.number}] "
                f"MRR={mrr:.4f} ({elapsed:.0f}s) | "
                f"dim={params['embedding_dim']}, "
                f"lr={params['learning_rate']:.5f}, "
                f"negs={params['num_negs_per_pos']}, "
                f"margin={params['loss_margin']:.2f}"
            )
            return mrr

        except (KeyboardInterrupt, SystemExit):
            # Re-raise interruption signals -- never let SLURM signals get swallowed
            raise
        except Exception as e:
            # Numerical instabilities (NaN, exploding gradients) etc.
            logger.warning(
                f"  [{model_name} trial {trial.number}] "
                f"FAILED: {type(e).__name__}: {e}"
            )
            raise optuna.TrialPruned()

    return objective


# --------------------------------------------------------------------------- #
# Custom evaluation (validated to match PyKEEN exactly in naive mode)
# --------------------------------------------------------------------------- #

def evaluate_custom(model, test_factory, train_factory, valid_factory,
                    type_index, mode, relation_label=RELATION_LABEL):
    """Custom evaluation supporting naive and type_constrained modes."""
    device = next(model.parameters()).device
    relation_id = train_factory.relation_to_id[relation_label]

    num_entities = train_factory.num_entities
    if mode == 'type_constrained':
        tail_mask = torch.full((num_entities,), float('-inf'))
        tail_mask[type_index['machine']] = 0.0
        head_mask = torch.full((num_entities,), float('-inf'))
        head_mask[type_index['item']] = 0.0
    else:
        tail_mask = torch.zeros(num_entities)
        head_mask = torch.zeros(num_entities)

    tail_mask = tail_mask.to(device)
    head_mask = head_mask.to(device)

    head_to_known_tails, tail_to_known_heads = {}, {}
    for fac in [train_factory, valid_factory, test_factory]:
        for h, r, t in fac.mapped_triples.tolist():
            if r != relation_id:
                continue
            head_to_known_tails.setdefault(h, set()).add(t)
            tail_to_known_heads.setdefault(t, set()).add(h)

    test_triples = test_factory.mapped_triples
    test_triples = test_triples[test_triples[:, 1] == relation_id]

    model.eval()
    tail_ranks, head_ranks = [], []

    with torch.no_grad():
        for h, r, t in test_triples.tolist():
            scores = model.score_t(torch.tensor([[h, r]], device=device)).squeeze(0)
            scores = scores + tail_mask
            for kt in head_to_known_tails.get(h, set()):
                if kt != t:
                    scores[kt] = float('-inf')
            tail_ranks.append((scores > scores[t]).sum().item() + 1)

            scores = model.score_h(torch.tensor([[r, t]], device=device)).squeeze(0)
            scores = scores + head_mask
            for kh in tail_to_known_heads.get(t, set()):
                if kh != h:
                    scores[kh] = float('-inf')
            head_ranks.append((scores > scores[h]).sum().item() + 1)

    def metrics(ranks):
        ranks = torch.tensor(ranks).float()
        m = {
            'MRR': (1.0 / ranks).mean().item(),
            'MR': ranks.mean().item(),
        }
        for k in HITS_AT_K:
            m[f'Hits@{k}'] = (ranks <= k).float().mean().item()
        return m

    tail_m = metrics(tail_ranks)
    head_m = metrics(head_ranks)
    avg_m = {k: (tail_m[k] + head_m[k]) / 2 for k in tail_m}
    return {'tail': tail_m, 'head': head_m, 'avg': avg_m}


def retrain_and_evaluate_best(
    model_name, best_params,
    train_factory, valid_factory, test_factory, type_index, device,
):
    """Retrain with best hyperparameters (early stopping enabled) and
    evaluate on the TEST set in both modes."""
    logger.info(f"  Retraining {model_name} with best params, "
                f"up to {TRAINING_EPOCHS_FINAL} epochs (early stopping)...")

    kwargs = build_pipeline_kwargs(
        model_name, best_params, TRAINING_EPOCHS_FINAL, device,
        use_early_stopping=True,
    )
    result = pipeline(
        training=train_factory,
        validation=valid_factory,
        testing=valid_factory,
        **kwargs,
    )

    naive_results = evaluate_custom(
        result.model, test_factory, train_factory, valid_factory,
        type_index, mode='naive',
    )
    typed_results = evaluate_custom(
        result.model, test_factory, train_factory, valid_factory,
        type_index, mode='type_constrained',
    )

    return result.model, {
        'naive': naive_results,
        'type_constrained': typed_results,
    }


# --------------------------------------------------------------------------- #
# Main pipeline per model
# --------------------------------------------------------------------------- #

def optimize_model(model_name, train_factory, valid_factory, test_factory,
                   type_index, device, n_trials, timeout_seconds):
    logger.info("\n" + "#" * 70)
    logger.info(f"# HPO FOR {model_name}")
    logger.info("#" * 70)

    out_dir = os.path.join(RESULTS_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)

    attach_file_logger(os.path.join(out_dir, 'run.log'))

    storage_path = os.path.join(out_dir, 'study.db')
    storage_url = f'sqlite:///{os.path.abspath(storage_path)}'

    study = optuna.create_study(
        study_name=f'kge_{model_name.lower()}',
        storage=storage_url,
        direction='maximize',
        sampler=TPESampler(seed=RANDOM_SEED),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        load_if_exists=True,
    )

    n_done = sum(
        1 for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    )
    n_remaining = max(0, n_trials - n_done)
    logger.info(f"  Completed trials so far : {n_done}")
    logger.info(f"  Trials still to run     : {n_remaining}")
    logger.info(f"  Time budget for HPO     : {timeout_seconds/3600:.1f} hours")

    if n_remaining > 0:
        objective = make_objective(model_name, train_factory, valid_factory, device)
        study.optimize(
            objective,
            n_trials=n_remaining,
            timeout=timeout_seconds,
            gc_after_trial=True,
        )

    n_done_after = sum(
        1 for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    )
    logger.info(f"\n  HPO finished. Completed trials: {n_done_after}")

    if n_done_after == 0:
        logger.error(f"  No trial completed for {model_name}. Skipping retrain.")
        return None, None

    logger.info(f"  Best MRR (validation): {study.best_value:.4f}")
    logger.info(f"  Best params: {study.best_params}")

    with open(os.path.join(out_dir, 'best_params.json'), 'w') as f:
        json.dump(study.best_params, f, indent=2)
    study.trials_dataframe().to_csv(
        os.path.join(out_dir, 'trials.csv'), index=False
    )

    logger.info(f"\n  Retraining {model_name} with best hyperparameters "
                f"and evaluating on TEST set...")
    final_model, test_metrics = retrain_and_evaluate_best(
        model_name, study.best_params,
        train_factory, valid_factory, test_factory, type_index, device,
    )

    torch.save(final_model.state_dict(), os.path.join(out_dir, 'best_model.pt'))
    with open(os.path.join(out_dir, 'best_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)

    logger.info(f"\n  === {model_name} TEST results ===")
    for mode in ['naive', 'type_constrained']:
        avg = test_metrics[mode]['avg']
        logger.info(
            f"  {mode:18s} | MRR={avg['MRR']:.4f} "
            f"H@1={avg['Hits@1']:.4f} "
            f"H@3={avg['Hits@3']:.4f} "
            f"H@10={avg['Hits@10']:.4f}"
        )

    return test_metrics, study.best_params


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=MODELS_TO_OPTIMIZE)
    parser.add_argument('--n_trials', type=int, default=N_TRIALS_PER_MODEL)
    parser.add_argument('--timeout_hours', type=float,
                        default=HPO_TIMEOUT_PER_MODEL / 3600,
                        help='Time budget per model (hours)')
    args = parser.parse_args()

    timeout_seconds = int(args.timeout_hours * 3600)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    logger.info("=" * 70)
    logger.info("STEP 2 (HPO): Hyperparameter Optimization for KGE models")
    logger.info("=" * 70)
    logger.info(f"  Models       : {args.models}")
    logger.info(f"  Trials/model : {args.n_trials}")
    logger.info(f"  HPO epochs   : {TRAINING_EPOCHS_HPO}")
    logger.info(f"  Final epochs : {TRAINING_EPOCHS_FINAL} (early stop enabled)")
    logger.info(f"  Time/model   : {args.timeout_hours:.1f} h")
    logger.info(f"  Output       : {RESULTS_DIR}")

    cuda_available = torch.cuda.is_available()
    device = 'cuda' if cuda_available else 'cpu'
    logger.info(f"  Device       : {device}")
    if cuda_available:
        logger.info(f"  GPU          : {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("  CUDA not available -- HPO on CPU will be VERY slow.")

    train_factory, valid_factory, test_factory = load_triples_factories()
    type_index = build_type_index(train_factory)

    summary = {}
    for model_name in args.models:
        try:
            metrics, best_params = optimize_model(
                model_name, train_factory, valid_factory, test_factory,
                type_index, device, args.n_trials, timeout_seconds,
            )
            if metrics is not None:
                summary[model_name] = {
                    'best_params': best_params,
                    'test_metrics': metrics,
                }
            else:
                summary[model_name] = {'error': 'no completed trials'}
        except (KeyboardInterrupt, SystemExit):
            logger.warning(f"  Interrupt received during {model_name}, exiting cleanly.")
            raise
        except Exception as e:
            logger.error(f"  HPO failed for {model_name}: {type(e).__name__}: {e}")
            summary[model_name] = {'error': str(e)}

    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("FINAL COMPARISON TABLE (TEST set, averaged head+tail)")
    logger.info("=" * 70)
    header = (f"{'Model':10s} {'Mode':18s} {'MRR':>8s} "
              f"{'H@1':>8s} {'H@3':>8s} {'H@10':>8s}")
    logger.info(header)
    logger.info("-" * len(header))
    for model_name in args.models:
        if 'error' in summary.get(model_name, {}):
            logger.info(f"{model_name:10s}  -- failed --")
            continue
        for mode in ['naive', 'type_constrained']:
            avg = summary[model_name]['test_metrics'][mode]['avg']
            logger.info(
                f"{model_name:10s} {mode:18s} "
                f"{avg['MRR']:>8.4f} {avg['Hits@1']:>8.4f} "
                f"{avg['Hits@3']:>8.4f} {avg['Hits@10']:>8.4f}"
            )

    logger.info("\nDONE.")


if __name__ == "__main__":
    main()
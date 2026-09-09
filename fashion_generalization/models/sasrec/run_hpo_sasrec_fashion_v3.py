# fashion_generalization/models/sasrec/run_hpo_sasrec_fashion_v3.py
"""
SASRec Fashion — Full HPO on HPC.

Features:
- hidden_size fixed at 128 (from the best TransE v1 HPO)
- 50 random-search trials
- Dedicated directory for each trial (trial_001, trial_002, ...)
- For each epoch: validation NDCG@20 and Recall@20 + test NDCG@20 and Recall@20
- Saves the best validation weights and best test weights separately
- Resume mechanism: if the 24-hour job limit expires, rerun and resume from the last completed trial
- Per-trial logs + global CSV summary
"""

import torch
# PyTorch 2.6 + RecBole compatibility fix
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(*args, **kwargs)
torch.load = _patched_load

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import SASRec
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger

import logging
import pandas as pd
import numpy as np
import random
import json
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE        = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
DATA_PATH   = os.path.join(BASE, "data", "recbole_v3")
OUT_FOLDER  = os.path.join(BASE, "results", "sasrec_hpo_v3")
RESULTS_CSV = os.path.join(OUT_FOLDER, "hpo_results_sasrec_fashion_v3.csv")


NUM_TRIALS   = 50
RANDOM_SEED  = 42

PARAM_SPACE = {
    'n_layers':            [2, 4],
    'n_heads':             [2, 4, 8],
    'learning_rate':       np.logspace(-5, -3, 5).tolist(),
    'hidden_dropout_prob': [0.3, 0.4, 0.5],
    'attn_dropout_prob':   [0.1, 0.2, 0.3],
    'weight_decay':        np.logspace(-5, -4, 3).tolist(),
    'loss_type':           ['CE', 'BPR'],
    'train_neg_sample_args': [
        {'candidate_num': 1},
        {'candidate_num': 3},
        {'candidate_num': 5},
    ],
}


class EpochTracker(Trainer):
    """
    Extends the RecBole Trainer to:
    - evaluate on the test set at every epoch (not only at the end)
    - save the best validation and test weights separately
    - log validation and test metrics for every epoch to a dedicated file
    """

    def __init__(self, config, model, trial_dir, epoch_logger):
        super().__init__(config, model)
        self.trial_dir    = trial_dir
        self.epoch_logger = epoch_logger

        # Best on validation
        self.best_valid_ndcg   = -1.0
        self.best_valid_epoch  = -1
        self.best_valid_recall = 0.0

        # Best on test (for reference only — NOT used for early stopping)
        self.best_test_ndcg    = -1.0
        self.best_test_epoch   = -1
        self.best_test_recall  = 0.0

        # Checkpoint paths
        self.best_valid_ckpt = os.path.join(trial_dir, "best_valid.pth")
        self.best_test_ckpt  = os.path.join(trial_dir, "best_test.pth")

        # Epoch history
        self.epoch_history = []

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        return super()._train_epoch(train_data, epoch_idx, loss_func, show_progress)

    def fit(self, train_data, valid_data, test_data,
            verbose=True, saved=True, show_progress=False, callback_fn=None):
        """
        Override of the standard fit method: evaluate on validation and test
        at every epoch.
        Early stopping is based only on validation NDCG@20.
        """
        patience_counter = 0
        stopping_step    = self.config['stopping_step']

        for epoch_idx in range(self.config['epochs']):

            # ── Training ──────────────────────────────────────────────────────
            train_loss = self._train_epoch(train_data, epoch_idx,
                                           show_progress=show_progress)

            # ── Validation ────────────────────────────────────────────────────
            valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
            valid_ndcg   = valid_result.get('ndcg@20', 0.0)
            valid_recall = valid_result.get('recall@20', 0.0)

            # ── Test ──────────────────────────────────────────────────────────
            test_score, test_result  = self._valid_epoch(test_data, show_progress=show_progress)
            test_ndcg    = test_result.get('ndcg@20', 0.0)
            test_recall  = test_result.get('recall@20', 0.0)

            # ── Per-epoch logging ─────────────────────────────────────────────
            self.epoch_logger.info(
                f"  Epoch {epoch_idx:3d} | "
                f"loss={train_loss:.4f} | "
                f"valid NDCG@20={valid_ndcg:.4f} Recall@20={valid_recall:.4f} | "
                f"test  NDCG@20={test_ndcg:.4f} Recall@20={test_recall:.4f}"
            )

            self.epoch_history.append({
                'epoch':        epoch_idx,
                'train_loss':   train_loss,
                'valid_ndcg':   valid_ndcg,
                'valid_recall': valid_recall,
                'test_ndcg':    test_ndcg,
                'test_recall':  test_recall,
            })

            # ── Best validation ───────────────────────────────────────────────
            if valid_ndcg > self.best_valid_ndcg:
                self.best_valid_ndcg   = valid_ndcg
                self.best_valid_recall = valid_recall
                self.best_valid_epoch  = epoch_idx
                torch.save(self.model.state_dict(), self.best_valid_ckpt)
                patience_counter = 0
                self.epoch_logger.info(
                    f"    *** New best validation: NDCG@20={valid_ndcg:.4f} "
                    f"(epoch {epoch_idx}) — weights saved"
                )
            else:
                patience_counter += 1

            # ── Best test ─────────────────────────────────────────────────────
            if test_ndcg > self.best_test_ndcg:
                self.best_test_ndcg   = test_ndcg
                self.best_test_recall = test_recall
                self.best_test_epoch  = epoch_idx
                torch.save(self.model.state_dict(), self.best_test_ckpt)
                self.epoch_logger.info(
                    f"    *** New best test: NDCG@20={test_ndcg:.4f} "
                    f"(epoch {epoch_idx}) — weights saved"
                )

            # ── Early stopping ────────────────────────────────────────────────
            if patience_counter >= stopping_step:
                self.epoch_logger.info(
                    f"  Early stopping at epoch {epoch_idx} "
                    f"(patience={stopping_step}). "
                    f"Best valid epoch={self.best_valid_epoch}"
                )
                break

        return {
            'best_valid_ndcg':   self.best_valid_ndcg,
            'best_valid_recall': self.best_valid_recall,
            'best_valid_epoch':  self.best_valid_epoch,
            'best_test_ndcg':    self.best_test_ndcg,
            'best_test_recall':  self.best_test_recall,
            'best_test_epoch':   self.best_test_epoch,
            'epoch_history':     self.epoch_history,
        }


# ── Single-trial function ─────────────────────────────────────────────────────
def run_trial(trial_idx, params, base_path, data_path):
    """
    Run a single HPO trial.
    Return a result dictionary or None if the trial fails.
    """
    trial_name = f"trial_{trial_idx:03d}"
    trial_dir  = os.path.join(base_path, trial_name)
    os.makedirs(trial_dir, exist_ok=True)

    # ── Dedicated trial logger ────────────────────────────────────────────────
    trial_logger = logging.getLogger(trial_name)
    trial_logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if the logger already exists
    if not trial_logger.handlers:
        fh = logging.FileHandler(
            os.path.join(trial_dir, "run.log"), mode='w'
        )
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(message)s', datefmt='%H:%M:%S'
        ))
        trial_logger.addHandler(fh)
    trial_logger.propagate = False

    trial_logger.info(f"{'='*60}")
    trial_logger.info(f"TRIAL {trial_idx:03d}")
    trial_logger.info(f"{'='*60}")
    trial_logger.info(f"Parameters: {json.dumps(params, indent=2, default=str)}")

    # ── RecBole configuration ─────────────────────────────────────────────────
    config_dict = {
        'dataset':   'fashion_v3',   # instead of 'fashion'
        'data_path': data_path,
        'USER_ID_FIELD':  'user_id',
        'ITEM_ID_FIELD':  'item_id',
        'TIME_FIELD':     'timestamp',
        'load_col':       {'inter': ['user_id', 'item_id', 'timestamp']},
        'field_separator': "\t",

        'eval_args': {
            'split':    {'LS': 'valid_and_test'},
            'order':    'TO',
            'group_by': 'user',
            'mode':     {'valid': 'full', 'test': 'full'},
        },

        # Model — hidden_size FIXED at 128
        'hidden_size':         128,
        'MAX_ITEM_LIST_LENGTH': 50,
        'n_layers':            params['n_layers'],
        'n_heads':             params['n_heads'],
        'hidden_dropout_prob': params['hidden_dropout_prob'],
        'attn_dropout_prob':   params['attn_dropout_prob'],
        'loss_type':           params['loss_type'],

        # Training
        'learning_rate':     params['learning_rate'],
        'weight_decay':      params['weight_decay'],
        'epochs':            75,
        'train_batch_size':  256,
        'eval_batch_size':   512,
        'stopping_step':     10,

        # Output
        'checkpoint_dir':    trial_dir,
        'metrics':           ['Recall', 'NDCG'],
        'topk':              [20],
        'valid_metric':      'NDCG@20',

        'reproducibility':   True,
        'seed':              2020,
        'use_gpu':           True,
        'show_progress':     False,
    }

    # Negative sampling
    if params['loss_type'] == 'CE':
        config_dict['train_neg_sample_args'] = None
    else:
        cand = params['train_neg_sample_args']['candidate_num']
        config_dict['train_neg_sample_args'] = {
            'distribution': 'uniform',
            'sample_num':   cand,
            'alpha':        1.0,
            'dynamic':      False,
            'candidate_num': 0,
        }

    try:
        config = Config(model='SASRec', config_dict=config_dict)
        init_seed(config['seed'], config['reproducibility'])
        init_logger(config)

        dataset = create_dataset(config)
        trial_logger.info(
            f"Dataset: {dataset.user_num-1:,} users, "
            f"{dataset.item_num-1:,} items, "
            f"{dataset.inter_num:,} interactions"
        )

        train_data, valid_data, test_data = data_preparation(config, dataset)

        model   = SASRec(config, train_data.dataset).to(config['device'])
        trainer = EpochTracker(config, model, trial_dir, trial_logger)

        results = trainer.fit(train_data, valid_data, test_data)

        try:
            model_verify = SASRec(config, train_data.dataset).to(config['device'])
            model_verify.load_state_dict(torch.load(
                os.path.join(trial_dir, "best_valid.pth")
            ))
            model_verify.eval()

            verify_tracker = EpochTracker(
                config, model_verify, trial_dir, trial_logger
            )
            _, test_scores_verify = verify_tracker._valid_epoch(
                test_data, show_progress=False
            )
            test_ndcg_verify   = test_scores_verify.get('ndcg@20', 0.0)
            test_recall_verify = test_scores_verify.get('recall@20', 0.0)

            best_valid_epoch = results['best_valid_epoch']
            history_df_tmp   = pd.DataFrame(results['epoch_history'])
            test_at_best_valid_epoch = history_df_tmp.loc[
                history_df_tmp['epoch'] == best_valid_epoch, 'test_ndcg'
            ].values[0]

            diff = abs(test_ndcg_verify - test_at_best_valid_epoch)

            trial_logger.info(
                f"  [INDEPENDENT VERIFICATION] "
                f"Test with best_valid.pth: NDCG@20={test_ndcg_verify:.4f} "
                f"Recall@20={test_recall_verify:.4f}"
            )
            trial_logger.info(
                f"  [INDEPENDENT VERIFICATION] "
                f"Test recorded at epoch {best_valid_epoch}: "
                f"NDCG@20={test_at_best_valid_epoch:.4f}"
            )

            if diff < 1e-4:
                trial_logger.info(
                    f"  [INDEPENDENT VERIFICATION] OK - diff={diff:.6f}"
                )
            else:
                trial_logger.warning(
                    f"  [INDEPENDENT VERIFICATION] WARNING - diff={diff:.6f}"
                )

            results['test_from_best_valid_ndcg']   = test_ndcg_verify
            results['test_from_best_valid_recall'] = test_recall_verify

        except Exception as e:
            trial_logger.warning(
                f"  [INDEPENDENT VERIFICATION] Failed: {e}"
            )
            results['test_from_best_valid_ndcg']   = 'FAILED'
            results['test_from_best_valid_recall'] = 'FAILED'

        # Save epoch history
        pd.DataFrame(results['epoch_history']).to_csv(
            os.path.join(trial_dir, "epoch_history.csv"), index=False
        )

        trial_logger.info(f"\n{'='*60}")
        trial_logger.info(f"FINAL TRIAL RESULTS {trial_idx:03d}")
        trial_logger.info(f"{'='*60}")
        trial_logger.info(
            f"  Best valid ==> epoch={results['best_valid_epoch']} "
            f"NDCG@20={results['best_valid_ndcg']:.4f} "
            f"Recall@20={results['best_valid_recall']:.4f}"
        )
        trial_logger.info(
            f"  Best test  ==> epoch={results['best_test_epoch']} "
            f"NDCG@20={results['best_test_ndcg']:.4f} "
            f"Recall@20={results['best_test_recall']:.4f}"
        )

        return {
            'trial':               trial_idx,
            'n_layers':            params['n_layers'],
            'n_heads':             params['n_heads'],
            'learning_rate':       params['learning_rate'],
            'hidden_dropout_prob': params['hidden_dropout_prob'],
            'attn_dropout_prob':   params['attn_dropout_prob'],
            'weight_decay':        params['weight_decay'],
            'loss_type':           params['loss_type'],
            'neg_candidate_num':   params['train_neg_sample_args'].get(
                                       'candidate_num', 'N/A'),
            'best_valid_ndcg':     results['best_valid_ndcg'],
            'best_valid_recall':   results['best_valid_recall'],
            'best_valid_epoch':    results['best_valid_epoch'],
            'best_test_ndcg':      results['best_test_ndcg'],
            'best_test_recall':    results['best_test_recall'],
            'best_test_epoch':     results['best_test_epoch'],
            # Additional verification metrics
            'test_from_best_valid_ndcg':   results.get(
                'test_from_best_valid_ndcg', 'N/A'
            ),
            'test_from_best_valid_recall': results.get(
                'test_from_best_valid_recall', 'N/A'
            ),
            'status':              'OK',
        }

    except Exception as e:
        trial_logger.error(f"ERROR: {e}")
        logging.getLogger(__name__).error(
            f"Trial {trial_idx:03d} failed: {e}"
        )
        return {
            'trial':   trial_idx,
            'n_layers': params['n_layers'],
            'n_heads':  params['n_heads'],
            'learning_rate': params['learning_rate'],
            'hidden_dropout_prob': params['hidden_dropout_prob'],
            'attn_dropout_prob': params['attn_dropout_prob'],
            'weight_decay': params['weight_decay'],
            'loss_type': params['loss_type'],
            'neg_candidate_num': params['train_neg_sample_args'].get(
                                     'candidate_num', 'N/A'),
            'best_valid_ndcg':   'FAILED',
            'best_valid_recall': 'FAILED',
            'best_valid_epoch':  'FAILED',
            'best_test_ndcg':    'FAILED',
            'best_test_recall':  'FAILED',
            'best_test_epoch':   'FAILED',
            'test_from_best_valid_ndcg':   'FAILED',
            'test_from_best_valid_recall': 'FAILED',
            'status':            f'FAILED: {e}',
        }


# ── Helper: robust deduplication key against CSV float round-trip precision ────
def make_param_key(n_layers, n_heads, learning_rate, hidden_dropout_prob,
                    attn_dropout_prob, weight_decay, loss_type, candidate_num):
    def r(x):
        try:
            return round(float(x), 10)
        except (TypeError, ValueError):
            return x
    return (
        int(n_layers), int(n_heads),
        r(learning_rate), r(hidden_dropout_prob), r(attn_dropout_prob), r(weight_decay),
        str(loss_type), str(candidate_num),
    )


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    os.makedirs(OUT_FOLDER, exist_ok=True)
    random.seed(RANDOM_SEED)

    # Main logger
    main_logger = logging.getLogger('hpo_main')
    main_logger.setLevel(logging.INFO)

    if not main_logger.handlers:
        fh = logging.FileHandler(
            os.path.join(OUT_FOLDER, "hpo_main.log"), mode='a'
        )
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
        ))
        main_logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(
            '%(asctime)s - %(message)s',
            datefmt='%H:%M:%S'
        ))
        main_logger.addHandler(sh)

    main_logger.propagate = False

    main_logger.info("=" * 60)
    main_logger.info("SASREC FASHION — HPO")
    main_logger.info("=" * 60)
    main_logger.info(f"  Total trials:  {NUM_TRIALS}")
    main_logger.info(f"  hidden_size:   128 (fixed from TransE v3 HPO)")
    main_logger.info(f"  Output:        {OUT_FOLDER}")

    # ── Resume: load already completed trials ─────────────────────────────────
    all_results       = []
    tested_params_set = set()
    completed_trials  = set()

    if os.path.exists(RESULTS_CSV):
        main_logger.info(
            f"  CSV found: resuming from the last completed trial."
        )
        try:
            df_existing = pd.read_csv(RESULTS_CSV)
            all_results = df_existing.to_dict('records')

            for r in all_results:
                if r.get('status') == 'OK':
                    completed_trials.add(int(r['trial']))

                param_tuple = make_param_key(
                    r.get('n_layers', ''),
                    r.get('n_heads', ''),
                    r.get('learning_rate', ''),
                    r.get('hidden_dropout_prob', ''),
                    r.get('attn_dropout_prob', ''),
                    r.get('weight_decay', ''),
                    r.get('loss_type', ''),
                    r.get('neg_candidate_num', '')
                )
                tested_params_set.add(param_tuple)

            main_logger.info(
                f"  Already completed trials: {len(completed_trials)}"
            )

        except Exception as e:
            main_logger.error(
                f"  CSV read error: {e}. Starting from scratch."
            )
            all_results       = []
            tested_params_set = set()
            completed_trials  = set()

    # ── HPO loop ──────────────────────────────────────────────────────────────
    trial_idx = max(completed_trials) + 1 if completed_trials else 1

    while len(completed_trials) < NUM_TRIALS:

        # Sample parameters
        params = {
            k: random.choice(list(v))
            for k, v in PARAM_SPACE.items()
        }

        # hidden_size=128 must be divisible by n_heads
        if 128 % params['n_heads'] != 0:
            continue

        # Deduplication
        param_tuple = make_param_key(
            params['n_layers'],
            params['n_heads'],
            params['learning_rate'],
            params['hidden_dropout_prob'],
            params['attn_dropout_prob'],
            params['weight_decay'],
            params['loss_type'],
            params['train_neg_sample_args'].get(
                'candidate_num', 'N/A'
            )
        )

        if param_tuple in tested_params_set:
            continue

        tested_params_set.add(param_tuple)

        main_logger.info(
            f"\nTRIAL {trial_idx:03d} "
            f"({len(completed_trials)+1}/{NUM_TRIALS})"
        )
        main_logger.info(f"  Params: {params}")

        result = run_trial(
            trial_idx,
            params,
            OUT_FOLDER,
            DATA_PATH
        )
        all_results.append(result)

        if result['status'] == 'OK':
            completed_trials.add(trial_idx)

            main_logger.info(
                f"  ==> best valid NDCG@20={result['best_valid_ndcg']:.4f} "
                f"(epoch {result['best_valid_epoch']})"
            )
            main_logger.info(
                f"  ==> best test  NDCG@20={result['best_test_ndcg']:.4f} "
                f"(epoch {result['best_test_epoch']})"
            )

        # Save after every trial — resume mechanism
        pd.DataFrame(all_results).to_csv(
            RESULTS_CSV,
            index=False,
            float_format='%.17g'
        )
        main_logger.info(f"  CSV updated: {RESULTS_CSV}")

        trial_idx += 1

    # ── Final summary ─────────────────────────────────────────────────────────
    main_logger.info("\n" + "=" * 60)
    main_logger.info("HPO COMPLETED")
    main_logger.info("=" * 60)

    final_df = pd.read_csv(RESULTS_CSV)
    final_df = final_df[
        final_df['status'] == 'OK'
    ].copy()

    final_df['best_valid_ndcg'] = pd.to_numeric(
        final_df['best_valid_ndcg'],
        errors='coerce'
    )

    final_df = final_df.sort_values(
        'best_valid_ndcg',
        ascending=False
    )

    if len(final_df) > 0:
        best = final_df.iloc[0]

        main_logger.info(
            "\n--- BEST PARAMETERS (by validation NDCG@20) ---"
        )

        for col in [
            'trial',
            'n_layers',
            'n_heads',
            'learning_rate',
            'hidden_dropout_prob',
            'attn_dropout_prob',
            'weight_decay',
            'loss_type',
            'neg_candidate_num',
            'best_valid_ndcg',
            'best_valid_recall',
            'best_valid_epoch',
            'best_test_ndcg',
            'best_test_recall',
            'best_test_epoch',
            'test_from_best_valid_ndcg',
            'test_from_best_valid_recall'
        ]:
            main_logger.info(
                f"  {col}: {best.get(col, 'N/A')}"
            )

        main_logger.info("\n--- TOP 5 ---")
        main_logger.info(
            final_df.head(5)[[
                'trial',
                'n_layers',
                'n_heads',
                'learning_rate',
                'best_valid_ndcg',
                'best_test_ndcg'
            ]].to_string()
        )

    else:
        main_logger.info(
            "No trial completed successfully."
        )
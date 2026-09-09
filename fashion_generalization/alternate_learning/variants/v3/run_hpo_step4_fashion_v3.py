"""
run_hpo_step4_fashion_v3.py
=============================

HPO wrapper for Fashion Step 4 v3 using Optuna TPE.

Hyperparameter optimization is performed only for --relations 3.
The selected hyperparameters are then reused for the 5-relation
configuration and the four ablation variants.

It follows the same logic as run_hpo_step4_fashion_ce.py (v1),
with the following differences:

  - Runs run_step4_fashion_v3.py with --relations fixed to 3.
  - HPO results are stored under hpo_results_fashion_v3/.
  - Step 4 results are read from:
        hpc_step4_results_v3/step4_3rel_{sched}_ce{run_tag}
  - The first enqueued trial uses margin_loss=8.95047357618049,
    corresponding to the v3 3-relation TransE configuration.
  - Each trial runs for 20 epochs by default instead of 50,
    to remain within the 24-hour SLURM limit while allowing
    multiple trials per job/resubmission.

Example on HPC:

  python alternate_learning/variants/v3/run_hpo_step4_fashion_v3.py \\
      --base-config alternate_learning/configs/step4_config_fashion_v3.yaml \\
      --trials 30 \\
      --epochs 20 \\
      --scheduling one_to_one \\
      --storage sqlite:///alternate_learning/hpo_results_fashion_v3/optuna_fashion_v3.db
"""

import os
import sys
import argparse
import yaml
import json
import csv
import shutil
import subprocess
import random
import time
from pathlib import Path

try:
    import optuna
except Exception as e:
    raise RuntimeError(
        "Optuna not found. Install it with: pip install optuna"
    ) from e

try:
    from optuna.trial import TrialState
except Exception:
    TrialState = None


THIS_FILE = Path(__file__).resolve()

# This file is stored under:
#   alternate_learning/variants/v3/
V3_DIR = THIS_FILE.parent

# Main alternate-learning directory:
#   alternate_learning/
PROJECT_ROOT = V3_DIR.parents[1]

DEFAULT_BASE_CONFIG = (
    PROJECT_ROOT
    / 'configs'
    / 'step4_config_fashion_v3.yaml'
)

DEFAULT_RESULTS_ROOT = (
    PROJECT_ROOT
    / 'hpo_results_fashion_v3'
)

DEFAULT_SEEDS  = [2020]
DEFAULT_TRIALS = 30

# Reduced from 50 epochs in v1 to remain within the 24-hour
# budget while allowing multiple trials per job.
DEFAULT_EPOCHS = 20

# HPO is performed only on the 3-relation configuration.
N_RELATIONS = 3


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def write_yaml(obj, path):
    with open(path, 'w') as f:
        yaml.safe_dump(obj, f)


def merge_config(base_cfg, overrides):
    cfg = dict(base_cfg)

    for k, v in overrides.items():
        if '.' in k:
            parts = k.split('.')
            cur = cfg

            for p in parts[:-1]:
                cur = cur.setdefault(p, {})

            cur[parts[-1]] = v
        else:
            cfg[k] = v

    return cfg


def run_trial_process(
    config_path,
    epochs,
    env=None,
    timeout=None
):
    """
    Run run_step4_fashion_v3.py as a subprocess,
    with --relations fixed to 3.
    """
    cmd = [
        sys.executable,
        str(V3_DIR / 'run_step4_fashion_v3.py'),
        '--config',
        str(config_path),
        '--relations',
        str(N_RELATIONS),
    ]

    if epochs is not None:
        cmd += [
            '--epochs',
            str(epochs)
        ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        encoding='utf-8',
        errors='replace',
        timeout=timeout,
    )

    return proc.returncode, proc.stdout


def append_skipped_trial(
    study_dir,
    trial_number,
    reason,
    config_path=None,
    seed=None,
    params=None,
    stdout_text=None
):
    skipped_path = os.path.join(
        study_dir,
        'skipped_configs.txt'
    )

    ts = time.strftime(
        '%Y-%m-%d %H:%M:%S'
    )

    with open(
        skipped_path,
        'a',
        encoding='utf-8'
    ) as f:

        f.write(
            f'[{ts}] trial={trial_number} skipped\n'
        )

        f.write(
            f'  reason: {reason}\n'
        )

        if config_path:
            f.write(
                f'  config: {config_path}\n'
            )

        if seed is not None:
            f.write(
                f'  seed: {seed}\n'
            )

        if params is not None:
            f.write(
                f'  params: '
                f'{json.dumps(params, ensure_ascii=False)}\n'
            )

        if stdout_text:
            f.write(
                '  stdout (first 80 lines):\n'
            )

            for line in stdout_text.splitlines()[:80]:
                f.write(
                    f'    {line}\n'
                )

        f.write('\n')


def params_signature(params):
    return tuple(
        sorted(
            (
                k,
                json.dumps(
                    v,
                    sort_keys=True,
                    ensure_ascii=False
                )
            )
            for k, v in params.items()
        )
    )


def find_duplicate_trial(
    study,
    params
):
    target = params_signature(
        params
    )

    states = []

    if TrialState is not None:
        states = [
            s
            for s in [
                TrialState.COMPLETE,
                TrialState.PRUNED
            ]
            if s is not None
        ]

    for t in study.get_trials(
        deepcopy=False,
        states=states or None
    ):
        if t.number is None:
            continue

        if params_signature(
            t.params
        ) == target:
            return t.number

    return None


def parse_fashion_log(out: str):
    """
    Extract metrics from the output of run_step4_fashion_v3.py.

    Expected line format:

    '==> Valid: R@20=0.0450 NDCG@20=0.0210 |
          Test: R@20=0.0380 NDCG@20=0.0170'
    """
    best_val_r20  = 0.0
    best_val_ndcg = 0.0

    last_test_r20  = None
    last_test_ndcg = None

    for line in out.splitlines():

        line_l = line.lower()

        if (
            '==>' in line_l
            and 'valid:' in line_l
            and 'r@20=' in line_l
        ):
            try:
                valid_part = (
                    line_l
                    .split('valid:')[1]
                    .split('|')[0]
                )

                val_r = float(
                    valid_part
                    .split('r@20=')[1]
                    .split()[0]
                )

                val_n = float(
                    valid_part
                    .split('ndcg@20=')[1]
                    .split()[0]
                )

                test_part = (
                    line_l
                    .split('test:')[1]
                )

                tst_r = float(
                    test_part
                    .split('r@20=')[1]
                    .split()[0]
                )

                tst_n = float(
                    test_part
                    .split('ndcg@20=')[1]
                    .split()[0]
                )

                if val_r > best_val_r20:
                    best_val_r20  = val_r
                    best_val_ndcg = val_n

                last_test_r20  = tst_r
                last_test_ndcg = tst_n

            except Exception:
                pass

    return (
        best_val_r20,
        best_val_ndcg,
        last_test_r20,
        last_test_ndcg
    )


def objective_factory(
    base_cfg,
    args,
    study_dir
):

    def objective(trial):

        try:
            lr = trial.suggest_float(
                'learning_rate',
                1e-5,
                5e-4,
                log=True
            )

            batch_size_sasrec = trial.suggest_categorical(
                'batch_size_sasrec',
                [
                    128,
                    256,
                    512,
                    1024
                ]
            )

            batch_size_kge = trial.suggest_categorical(
                'batch_size_kge',
                [
                    512,
                    1024,
                    2048
                ]
            )

            margin_a = trial.suggest_float(
                'margin_loss',
                0.5,
                10.0
            )

            hidden_dropout = trial.suggest_float(
                'hidden_dropout',
                0.1,
                0.6
            )

            attn_dropout = trial.suggest_float(
                'attn_dropout',
                0.0,
                0.6
            )

            seeds_list = getattr(
                args,
                '_seeds_list',
                DEFAULT_SEEDS
            )

            seed = int(
                seeds_list[
                    trial.number
                    % len(seeds_list)
                ]
            )

            run_tag = (
                f"_hpo_trial"
                f"{trial.number}"
                f"_seed{seed}"
            )

            overrides = {
                'training.learning_rate':
                    float(lr),

                'training.batch_size_sasrec':
                    int(batch_size_sasrec),

                'training.batch_size_kge':
                    int(batch_size_kge),

                'training.margin_loss':
                    float(margin_a),

                'training.scheduling':
                    args.scheduling,

                'training.run_tag':
                    run_tag,

                'model.hidden_dropout':
                    float(hidden_dropout),

                'model.attn_dropout':
                    float(attn_dropout),
            }

            recbole_cfg = dict(
                base_cfg.get(
                    'recbole_config',
                    {}
                )
            )

            recbole_cfg[
                'seed'
            ] = int(seed)

            overrides[
                'recbole_config'
            ] = recbole_cfg

            duplicate_params = {
                'training.learning_rate':
                    float(lr),

                'training.batch_size_sasrec':
                    int(batch_size_sasrec),

                'training.batch_size_kge':
                    int(batch_size_kge),

                'training.margin_loss':
                    float(margin_a),

                'model.hidden_dropout':
                    float(hidden_dropout),

                'model.attn_dropout':
                    float(attn_dropout),

                'training.scheduling':
                    args.scheduling,
            }

            dup = find_duplicate_trial(
                trial.study,
                duplicate_params
            )

            if dup is not None:

                append_skipped_trial(
                    study_dir=study_dir,
                    trial_number=trial.number,
                    reason=(
                        f'duplicate of trial {dup}'
                    ),
                    params=duplicate_params,
                    seed=seed,
                )

                raise optuna.exceptions.TrialPruned()

            trial_cfg = merge_config(
                base_cfg,
                overrides
            )

            os.makedirs(
                study_dir,
                exist_ok=True
            )

            cfg_path = os.path.join(
                study_dir,
                f'config_trial_{trial.number}.yaml'
            )

            write_yaml(
                trial_cfg,
                cfg_path
            )

            env = os.environ.copy()

            env[
                'PYTHONHASHSEED'
            ] = str(seed)

            env[
                'HPO_TRIAL'
            ] = str(trial.number)

            env[
                'PYTHONIOENCODING'
            ] = 'utf-8'

            retcode, out = run_trial_process(
                cfg_path,
                args.epochs,
                env=env,
                timeout=args.timeout
            )

            if retcode != 0:

                append_skipped_trial(
                    study_dir=study_dir,
                    trial_number=trial.number,
                    reason=(
                        f'run_step4_fashion_v3 '
                        f'exitcode={retcode}'
                    ),
                    config_path=cfg_path,
                    seed=seed,
                    params=trial.params,
                    stdout_text=out,
                )

                raise optuna.exceptions.TrialPruned()

        except subprocess.TimeoutExpired:

            append_skipped_trial(
                study_dir=study_dir,
                trial_number=trial.number,
                reason='timeout',
                config_path=locals().get(
                    'cfg_path'
                ),
                seed=locals().get(
                    'seed'
                ),
                params=trial.params,
            )

            raise optuna.exceptions.TrialPruned()

        except optuna.exceptions.TrialPruned:
            raise

        except Exception as exc:

            append_skipped_trial(
                study_dir=study_dir,
                trial_number=trial.number,
                reason=(
                    f'unexpected error: {exc}'
                ),
                config_path=locals().get(
                    'cfg_path'
                ),
                seed=locals().get(
                    'seed'
                ),
                params=trial.params,
            )

            raise optuna.exceptions.TrialPruned()

        stdout_path = os.path.join(
            study_dir,
            f'trial_{trial.number}_stdout.txt'
        )

        with open(
            stdout_path,
            'w',
            encoding='utf-8'
        ) as f:
            f.write(out)

        (
            best_val_r20,
            best_val_ndcg,
            last_test_r20,
            last_test_ndcg
        ) = parse_fashion_log(out)

        if best_val_r20 == 0.0:

            append_skipped_trial(
                study_dir=study_dir,
                trial_number=trial.number,
                reason=(
                    'validation Recall@20=0 — '
                    'no epoch was evaluated correctly'
                ),
                config_path=cfg_path,
                seed=seed,
                params=trial.params,
            )

            raise optuna.exceptions.TrialPruned()

        result = best_val_r20

        metrics = {
            'best_val_recall@20':
                best_val_r20,

            'best_val_ndcg@20':
                best_val_ndcg,

            'last_test_recall@20':
                last_test_r20,

            'last_test_ndcg@20':
                last_test_ndcg,
        }

        sched = trial_cfg[
            'training'
        ][
            'scheduling'
        ]

        run_tag_actual = trial_cfg[
            'training'
        ].get(
            'run_tag',
            ''
        )

        # Step 4 v3 output pattern:
        # hpc_step4_results_v3/
        #   step4_{n}rel_{sched}_ce{tag}/
        best_metrics_path = (
            PROJECT_ROOT
            / 'hpc_step4_results_v3'
            / (
                f"step4_"
                f"{N_RELATIONS}rel_"
                f"{sched}_ce"
                f"{run_tag_actual}"
            )
            / 'best_metrics.json'
        )

        best_epoch_valid = None
        best_epoch_test  = None
        best_test_r20    = None
        best_test_ndcg   = None

        test_from_best_valid_r20  = None
        test_from_best_valid_ndcg = None

        if best_metrics_path.exists():

            try:
                with open(
                    best_metrics_path,
                    encoding='utf-8'
                ) as f:
                    bm = json.load(f)

                best_epoch_valid = bm.get(
                    'best_epoch_valid'
                )

                best_epoch_test = bm.get(
                    'best_epoch_test'
                )

                best_test_r20 = bm.get(
                    'best_test_recall_20'
                )

                best_test_ndcg = bm.get(
                    'best_test_ndcg_20'
                )

                test_from_best_valid_r20 = bm.get(
                    'test_from_best_valid_recall'
                )

                test_from_best_valid_ndcg = bm.get(
                    'test_from_best_valid_ndcg'
                )

            except Exception:
                pass

        trial_dir = os.path.join(
            study_dir,
            f'trial_{trial.number}'
        )

        os.makedirs(
            trial_dir,
            exist_ok=True
        )

        meta = {
            'trial':
                trial.number,

            'params':
                trial.params,

            'seed':
                seed,

            'result':
                float(result),

            'metrics':
                metrics,

            'best_epoch_valid':
                best_epoch_valid,

            'best_epoch_test':
                best_epoch_test,

            'test_from_best_valid_recall':
                test_from_best_valid_r20,

            'test_from_best_valid_ndcg':
                test_from_best_valid_ndcg,
        }

        with open(
            os.path.join(
                trial_dir,
                f'trial_{trial.number}_meta.json'
            ),
            'w',
            encoding='utf-8'
        ) as f:

            json.dump(
                meta,
                f,
                indent=2,
                ensure_ascii=False
            )

        # Source directory containing the Step 4 v3 outputs
        src_folder = (
            PROJECT_ROOT
            / 'hpc_step4_results_v3'
            / (
                f"step4_"
                f"{N_RELATIONS}rel_"
                f"{sched}_ce"
                f"{run_tag_actual}"
            )
        )

        dest_outputs = os.path.join(
            trial_dir,
            'outputs'
        )

        try:

            if src_folder.exists():

                if os.path.exists(
                    dest_outputs
                ):
                    shutil.rmtree(
                        dest_outputs
                    )

                os.makedirs(
                    dest_outputs,
                    exist_ok=True
                )

                for fname in [
                    'best_model.pt',
                    'best_model_on_test.pt',
                    'last_model.pt',
                    'best_metrics.json',
                    'training_history.json',
                    'run.log'
                ]:

                    src_file = (
                        src_folder
                        / fname
                    )

                    if src_file.exists():
                        shutil.copy(
                            src_file,
                            os.path.join(
                                dest_outputs,
                                fname
                            )
                        )

                for ckpt in src_folder.glob(
                    'checkpoint_latest.pth'
                ):
                    try:
                        ckpt.unlink()
                    except Exception:
                        pass

        except Exception:
            pass

        trials_csv = os.path.join(
            study_dir,
            'trials_summary.csv'
        )

        ts = int(
            time.time()
        )

        with open(
            trials_csv,
            'a',
            newline='',
            encoding='utf-8'
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                trial.number,
                ts,
                float(result),

                best_val_r20,
                best_val_ndcg,

                last_test_r20,
                last_test_ndcg,

                test_from_best_valid_r20,
                test_from_best_valid_ndcg,

                seed,
                lr,
                batch_size_sasrec,
                batch_size_kge,

                margin_a,
                hidden_dropout,
                attn_dropout,

                json.dumps(
                    metrics,
                    ensure_ascii=False
                ),

                trial_dir,
            ])

        best_valid_csv = os.path.join(
            study_dir,
            'best_valid.csv'
        )

        with open(
            best_valid_csv,
            'a',
            newline='',
            encoding='utf-8'
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                trial.number,
                best_epoch_valid,

                lr,
                batch_size_sasrec,
                batch_size_kge,

                margin_a,
                hidden_dropout,
                attn_dropout,

                best_val_r20,
                best_val_ndcg,

                test_from_best_valid_r20,
                test_from_best_valid_ndcg,
            ])

        best_test_csv = os.path.join(
            study_dir,
            'best_test.csv'
        )

        with open(
            best_test_csv,
            'a',
            newline='',
            encoding='utf-8'
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                trial.number,
                best_epoch_test,

                lr,
                batch_size_sasrec,
                batch_size_kge,

                margin_a,
                hidden_dropout,
                attn_dropout,

                best_test_r20,
                best_test_ndcg,
            ])

        return float(result)

    return objective


def main():

    parser = argparse.ArgumentParser(
        description=(
            'HPO Step 4 Fashion v3 '
            '(Optuna TPE)'
        )
    )

    parser.add_argument(
        '--base-config',
        type=str,
        default=str(
            DEFAULT_BASE_CONFIG
        )
    )

    parser.add_argument(
        '--trials',
        type=int,
        default=DEFAULT_TRIALS
    )

    parser.add_argument(
        '--study-name',
        type=str,
        default=None
    )

    parser.add_argument(
        '--storage',
        type=str,
        default=None
    )

    parser.add_argument(
        '--scheduling',
        type=str,
        choices=[
            'one_to_one',
            'epoch'
        ],
        default='one_to_one'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=DEFAULT_EPOCHS
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None
    )

    parser.add_argument(
        '--seeds',
        type=str,
        default=None
    )

    parser.add_argument(
        '--timeout',
        type=int,
        default=None
    )

    parser.add_argument(
        '--study-dir',
        type=str,
        default=None
    )

    args = parser.parse_args()

    if args.study_name is None:
        args.study_name = (
            f"step4_fashion_v3_"
            f"{args.scheduling}_ce_"
            f"{time.strftime('%Y%m%d_%H%M%S')}"
        )

    if args.study_dir is None:
        args.study_dir = str(
            DEFAULT_RESULTS_ROOT
            / args.study_name
        )

    seeds_list = list(
        DEFAULT_SEEDS
    )

    if args.seeds:

        try:
            seeds_list = [
                int(x.strip())
                for x in args.seeds.split(',')
                if x.strip()
            ]

        except Exception:
            raise SystemExit(
                '--seeds must be a comma-separated list of integers'
            )

    elif args.seed is not None:
        seeds_list = [
            int(args.seed)
        ]

    base_cfg = load_yaml(
        args.base_config
    )

    study_dir = args.study_dir

    os.makedirs(
        study_dir,
        exist_ok=True
    )

    shutil.copy(
        args.base_config,
        os.path.join(
            study_dir,
            'base_config.yaml'
        )
    )

    if args.storage:
        storage_url = args.storage

    else:
        db_path = os.path.abspath(
            os.path.join(
                study_dir,
                f"optuna_{args.study_name}.db"
            )
        )

        storage_url = (
            f"sqlite:///{db_path}"
        )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction='maximize',
        load_if_exists=True,
    )

    if len(study.trials) == 0:

        study.enqueue_trial({
            'learning_rate':
                0.0001,

            'batch_size_sasrec':
                256,

            'batch_size_kge':
                1024,

            # TransE 3-relation v3 loss margin
            'margin_loss':
                8.95047357618049,

            'hidden_dropout':
                0.5,

            'attn_dropout':
                0.3,
        })

    trials_csv = os.path.join(
        study_dir,
        'trials_summary.csv'
    )

    best_valid_csv = os.path.join(
        study_dir,
        'best_valid.csv'
    )

    best_test_csv = os.path.join(
        study_dir,
        'best_test.csv'
    )

    if not os.path.exists(
        trials_csv
    ):

        with open(
            trials_csv,
            'w',
            newline='',
            encoding='utf-8'
        ) as f:

            csv.writer(f).writerow([
                'trial',
                'timestamp',
                'objective_value',

                'best_val_recall@20',
                'best_val_ndcg@20',

                'last_test_recall@20',
                'last_test_ndcg@20',

                'test_from_best_valid_recall',
                'test_from_best_valid_ndcg',

                'seed',
                'learning_rate',
                'batch_size_sasrec',
                'batch_size_kge',

                'margin_loss',
                'hidden_dropout',
                'attn_dropout',

                'metrics',
                'trial_dir',
            ])

    if not os.path.exists(
        best_valid_csv
    ):

        with open(
            best_valid_csv,
            'w',
            newline='',
            encoding='utf-8'
        ) as f:

            csv.writer(f).writerow([
                'trial',
                'best_epoch_valid',

                'learning_rate',
                'batch_size_sasrec',
                'batch_size_kge',

                'margin_loss',
                'hidden_dropout',
                'attn_dropout',

                'valid_recall@20',
                'valid_ndcg@20',

                'test_from_best_valid_recall',
                'test_from_best_valid_ndcg',
            ])

    if not os.path.exists(
        best_test_csv
    ):

        with open(
            best_test_csv,
            'w',
            newline='',
            encoding='utf-8'
        ) as f:

            csv.writer(f).writerow([
                'trial',
                'best_epoch_test',

                'learning_rate',
                'batch_size_sasrec',
                'batch_size_kge',

                'margin_loss',
                'hidden_dropout',
                'attn_dropout',

                'test_recall@20',
                'test_ndcg@20',
            ])

    args._seeds_list = seeds_list

    objective = objective_factory(
        base_cfg,
        args,
        study_dir
    )

    study.optimize(
        objective,
        n_trials=args.trials
    )

    print(
        'Fashion v3 HPO completed.'
    )

    try:
        print(
            'Best trial:',
            study.best_trial
        )

    except ValueError:
        print(
            'No completed trials '
            '(all trials were pruned or failed).'
        )


if __name__ == '__main__':
    main()
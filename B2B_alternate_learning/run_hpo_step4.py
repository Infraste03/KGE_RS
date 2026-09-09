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
    raise RuntimeError("Optuna is required for HPO. Install with `pip install optuna`.") from e

try:
    from optuna.trial import TrialState
except Exception:
    TrialState = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = PROJECT_ROOT / 'B2B_alternate_learning' / 'configs' / 'step4_hpc_config.yaml'
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / 'hpo_results'
DEFAULT_SEEDS = [2020]
DEFAULT_TRIALS = 30
DEFAULT_EPOCHS = 50


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def write_yaml(obj, path):
    with open(path, 'w') as f:
        yaml.safe_dump(obj, f)


def merge_config(base_cfg, overrides):
    # shallow merge for nested dicts used here
    cfg = dict(base_cfg)
    for k, v in overrides.items():
        # support dotted keys like 'training.learning_rate'
        if '.' in k:
            parts = k.split('.')
            cur = cfg
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = v
        else:
            cfg[k] = v
    return cfg


def run_trial_process(config_path, epochs, env=None, timeout=None):
    cmd = [sys.executable, str(Path(__file__).resolve().parent / 'run_step4.py'), '--config', config_path]
    if epochs is not None:
        cmd += ['--epochs', str(epochs)]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, timeout=timeout)
    return proc.returncode, proc.stdout


def append_skipped_trial(study_dir, trial_number, reason, config_path=None, seed=None, params=None, stdout_text=None):
    skipped_path = os.path.join(study_dir, 'skipped_configs.txt')
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(skipped_path, 'a', encoding='utf-8') as f:
        f.write(f'[{timestamp}] trial={trial_number} skipped\n')
        f.write(f'  reason: {reason}\n')
        if config_path:
            f.write(f'  config: {config_path}\n')
        if seed is not None:
            f.write(f'  seed: {seed}\n')
        if params is not None:
            f.write(f'  params: {json.dumps(params, ensure_ascii=False)}\n')
        if stdout_text:
            f.write('  stdout:\n')
            for line in stdout_text.splitlines()[:80]:
                f.write(f'    {line}\n')
        f.write('\n')


def params_signature(params):
    return tuple(sorted((key, json.dumps(value, sort_keys=True, ensure_ascii=False)) for key, value in params.items()))


def find_duplicate_trial(study, params):
    target_signature = params_signature(params)
    finished_states = []
    if TrialState is not None:
        finished_states = [state for state in [TrialState.COMPLETE, TrialState.PRUNED] if state is not None]
    for existing_trial in study.get_trials(deepcopy=False, states=finished_states or None):
        if existing_trial.number is None:
            continue
        if params_signature(existing_trial.params) == target_signature:
            return existing_trial.number
    return None


def objective_factory(base_cfg, args, study_dir):
    def objective(trial):
        try:
            # Suggest hyperparameters
            lr = trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
            batch_size_sasrec = trial.suggest_categorical('batch_size_sasrec', [128, 256, 512, 1024])
            batch_size_kge = trial.suggest_categorical('batch_size_kge', [512, 1024, 2048])
            num_neg_b = trial.suggest_categorical('num_neg_b', [1, 2, 4, 8])
            margin_a = trial.suggest_float('margin_loss', 0.5, 10.0)

            hidden_dropout = trial.suggest_float('hidden_dropout', 0.1, 0.6)
            attn_dropout = trial.suggest_float('attn_dropout', 0.0, 0.6)

            # Seed selection
            seeds_list = getattr(args, '_seeds_list', DEFAULT_SEEDS)
            seed = int(seeds_list[trial.number % len(seeds_list)])
            
            run_tag = f"_hpo_trial{trial.number}_seed{seed}"

            # Build per-trial config
            overrides = {
                'training.learning_rate': float(lr),
                'training.batch_size_sasrec': int(batch_size_sasrec),
                'training.batch_size_kge': int(batch_size_kge),
                'training.num_neg_b': int(num_neg_b),
                'training.margin_loss': float(margin_a),
                'training.scheduling': args.scheduling,
                'training.run_tag': run_tag,          
                'model.hidden_dropout': float(hidden_dropout),
                'model.attn_dropout': float(attn_dropout),
            }

            # Also set recbole seed inside recbole_config if present
            recbole_cfg = base_cfg.get('recbole_config', {})
            recbole_cfg['seed'] = int(seed)
            overrides['recbole_config'] = recbole_cfg

            duplicate_params = {
                'training.learning_rate': float(lr),
                'training.batch_size_sasrec': int(batch_size_sasrec),
                'training.batch_size_kge': int(batch_size_kge),
                'training.num_neg_b': int(num_neg_b),
                'training.margin_loss': float(margin_a),
                'model.hidden_dropout': float(hidden_dropout),
                'model.attn_dropout': float(attn_dropout),
                'training.scheduling': args.scheduling,
            }

            duplicate_trial_number = find_duplicate_trial(trial.study, duplicate_params)
            if duplicate_trial_number is not None:
                append_skipped_trial(
                    study_dir=study_dir,
                    trial_number=trial.number,
                    reason=f'duplicate hyperparameter set already tested in trial {duplicate_trial_number}',
                    config_path=None,
                    seed=seed,
                    params=duplicate_params,
                )
                raise optuna.exceptions.TrialPruned()

            trial_cfg = merge_config(base_cfg, overrides)

            # Write trial config
            os.makedirs(study_dir, exist_ok=True)
            cfg_path = os.path.join(study_dir, f'config_trial_{trial.number}.yaml')
            write_yaml(trial_cfg, cfg_path)

            # Prepare environment for subprocess to improve reproducibility
            env = os.environ.copy()
            env['PYTHONHASHSEED'] = str(seed)
            env['HPO_TRIAL'] = str(trial.number)

            # Run run_step4.py (may be long). Use epochs override from args or from trial
            epochs = args.epochs
            retcode, out = run_trial_process(cfg_path, epochs, env=env, timeout=args.timeout)
            if retcode != 0:
                append_skipped_trial(
                    study_dir=study_dir,
                    trial_number=trial.number,
                    reason=f'run_step4 exited with code {retcode}',
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
                reason='run_step4 timeout',
                config_path=locals().get('cfg_path'),
                seed=locals().get('seed'),
                params=trial.params,
            )
            raise optuna.exceptions.TrialPruned()
        except Exception as exc:
            append_skipped_trial(
                study_dir=study_dir,
                trial_number=trial.number,
                reason=f'unexpected error: {exc}',
                config_path=locals().get('cfg_path'),
                seed=locals().get('seed'),
                params=trial.params,
            )
            raise optuna.exceptions.TrialPruned()

        # Save stdout for inspection
        stdout_path = os.path.join(study_dir, f'trial_{trial.number}_stdout.txt')
        with open(stdout_path, 'w', encoding='utf-8') as f:
            f.write(out)

        # Parse metrics from the current trial stdout, so each trial uses its own run.
        parsed_validation_recall_at_20 = None
        parsed_validation_ndcg_at_20 = None
        parsed_test_recall_at_20 = None
        parsed_test_ndcg_at_20 = None
        best_validation_recall_at_20 = 0.0
        best_validation_ndcg_at_20 = 0.0
        
        
        for line in out.splitlines():
            line_lower = line.lower()
            if '[validation]' in line_lower and 'r@20=' in line_lower and 'ndcg@20=' in line_lower:
                try:
                    parts = line_lower.split('r@20=')[-1]
                    val_r = float(parts.split()[0])
                    val_n = float(line_lower.split('ndcg@20=')[-1].strip())
                    if val_r > best_validation_recall_at_20:
                        best_validation_recall_at_20 = val_r
                        best_validation_ndcg_at_20 = val_n
                except Exception:
                    pass
            if '[test]' in line_lower and 'r@20=' in line_lower and 'ndcg@20=' in line_lower:
                try:
                    parts = line_lower.split('r@20=')[-1]
                    parsed_test_recall_at_20 = float(parts.split()[0])
                    parsed_test_ndcg_at_20 = float(line_lower.split('ndcg@20=')[-1].strip())
                except Exception:
                    pass

        # Require that we have validation recall (we optimize validation Recall@20)
        if best_validation_recall_at_20 == 0.0:
            raise optuna.exceptions.TrialPruned()

        # objective = validation Recall@20 for this trial run (fair HPO)
        result = best_validation_recall_at_20

        # explicit fields for CSVs
        valid_recall_at_20 = best_validation_recall_at_20
        valid_ndcg_at_20 = best_validation_ndcg_at_20
        test_recall_at_20 = parsed_test_recall_at_20
        test_ndcg_at_20 = parsed_test_ndcg_at_20
        metrics = {
            'validation_recall@20': valid_recall_at_20,
            'validation_ndcg@20': valid_ndcg_at_20,
            'test_recall@20': test_recall_at_20,
            'test_ndcg@20': test_ndcg_at_20,
        }

        # Save trial metadata
        meta = {
            'trial': trial.number,
            'params': trial.params,
            'seed': seed,
            'result': float(result),
            'metrics': metrics,
        }
        trial_dir = os.path.join(study_dir, f'trial_{trial.number}')
        os.makedirs(trial_dir, exist_ok=True)
        with open(os.path.join(trial_dir, f'trial_{trial.number}_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
            
        # Leggi best_metrics.json per avere epoch e metriche corrette del best model
        best_metrics_path = Path('hpc_step4_results') / f"step4_{args.scheduling}{run_tag}" / 'best_metrics.json'
        best_epoch_valid = None
        best_epoch_test = None
        best_test_r20 = None
        best_test_ndcg = None
        if best_metrics_path.exists():
            try:
                with open(best_metrics_path) as f:
                    bm = json.load(f)
                best_epoch_valid = bm.get('best_epoch')
                best_epoch_test = bm.get('best_epoch_test')
                best_test_r20 = bm.get('R@20')
                best_test_ndcg = bm.get('NDCG@20')
            except Exception:
                pass
        # Copy outputs (if any) from the trial-specific scheduling folder to trial dir for inspection and checkpoints.
        # Using the run_tag guarantees isolation between trials.
        src_folder = Path('hpc_step4_results') / f"step4_{trial_cfg['training']['scheduling']}{trial_cfg['training'].get('run_tag','')}"
        dest_outputs = os.path.join(trial_dir, 'outputs')
        try:
            if src_folder.exists():
                # copytree may fail if dest exists; remove then copy
                if os.path.exists(dest_outputs):
                    shutil.rmtree(dest_outputs)
                shutil.copytree(src_folder, dest_outputs)

                # find model files
                model_files = list(Path(dest_outputs).rglob('*.pt')) + list(Path(dest_outputs).rglob('*.pth'))
                # copy deterministic filenames produced by run_step4.py
                best_valid_src = Path(src_folder) / 'best_model.pt'
                best_test_src = Path(src_folder) / 'best_model_on_test.pt'
                if best_valid_src.exists():
                    shutil.copy(best_valid_src, os.path.join(trial_dir, 'best_valid_model.pt'))
                if best_test_src.exists():
                    shutil.copy(best_test_src, os.path.join(trial_dir, 'best_test_model.pt'))

                # fallback: if names changed, keep the first .pt as valid and the second as test where possible
                if not best_valid_src.exists() or not best_test_src.exists():
                    pt_files = sorted(model_files, key=lambda p: p.stat().st_mtime)
                    if pt_files and not best_valid_src.exists():
                        shutil.copy(pt_files[0], os.path.join(trial_dir, 'best_valid_model.pt'))
                    if len(pt_files) > 1 and not best_test_src.exists():
                        shutil.copy(pt_files[-1], os.path.join(trial_dir, 'best_test_model.pt'))

        except Exception:
            # ignore copy errors but continue
            pass

        # Append to CSV summaries
        trials_csv = os.path.join(study_dir, 'trials_summary.csv')
        timestamp = int(time.time())
        with open(trials_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                trial.number,
                timestamp,
                float(result),
                valid_recall_at_20,
                valid_ndcg_at_20,
                test_recall_at_20,
                test_ndcg_at_20,
                seed,
                lr,
                batch_size_sasrec,
                batch_size_kge,
                num_neg_b,
                margin_a,
                hidden_dropout,
                attn_dropout,
                json.dumps(metrics),
                trial_dir
            ])

        # If we have explicit validation/test metrics, append to respective CSVs
        best_valid_csv = os.path.join(study_dir, 'best_valid.csv')
        best_test_csv = os.path.join(study_dir, 'best_test.csv')

        if valid_ndcg_at_20 is not None:
            with open(best_valid_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    trial.number,
                    best_epoch_valid,
                    lr,
                    batch_size_sasrec,
                    batch_size_kge,
                    num_neg_b,
                    margin_a,
                    hidden_dropout,
                    attn_dropout,
                    valid_recall_at_20,
                    valid_ndcg_at_20,
                ])

        if best_test_r20 is not None:
            with open(best_test_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    trial.number,
                    best_epoch_test,
                    lr,
                    batch_size_sasrec,
                    batch_size_kge,
                    num_neg_b,
                    margin_a,
                    hidden_dropout,
                    attn_dropout,
                    best_test_r20,
                    best_test_ndcg,
                ])

        return float(result)

    return objective


def main():
    parser = argparse.ArgumentParser(description='HPO wrapper for Step 4 (Optuna)')
    parser.add_argument('--base-config', type=str, default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument('--trials', type=int, default=DEFAULT_TRIALS)
    parser.add_argument('--study-name', type=str, default=None)
    parser.add_argument('--storage', type=str, default=None, help='Optuna storage URL (sqlite:///... or RDB)')
    parser.add_argument('--scheduling', type=str, choices=['one_to_one', 'epoch'], default='one_to_one')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Epochs to run per trial')
    parser.add_argument('--seed', type=int, default=None, help='Optional single seed override')
    parser.add_argument('--seeds', type=str, default=None, help='Optional comma-separated list of seeds override, e.g. "2024,199,42"')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout seconds for each trial subprocess')
    parser.add_argument('--study-dir', type=str, default=None)
    args = parser.parse_args()

    # study name and storage directory are generated automatically unless overridden
    if args.study_name is None:
        args.study_name = f"step4_{args.scheduling}_{time.strftime('%Y%m%d_%H%M%S')}"

    if args.study_dir is None:
        args.study_dir = str(DEFAULT_RESULTS_ROOT / args.study_name)

    # parse seeds override if provided; otherwise use the internal default list
    seeds_list = list(DEFAULT_SEEDS)
    if args.seeds:
        try:
            seeds_list = [int(x.strip()) for x in args.seeds.split(',') if x.strip()]
        except Exception:
            raise SystemExit('Invalid --seeds format. Use comma-separated integers, e.g. "2024,199,42"')
    elif args.seed is not None:
        seeds_list = [int(args.seed)]

    base_cfg = load_yaml(args.base_config)

    study_dir = args.study_dir
    os.makedirs(study_dir, exist_ok=True)
    # Save base config copy
    shutil.copy(args.base_config, os.path.join(study_dir, 'base_config.yaml'))

    # Create study (use sqlite in study_dir by default for resume across jobs)
    if args.storage:
        storage_url = args.storage
    else:
        db_path = os.path.abspath(os.path.join(study_dir, f"optuna_{args.study_name}.db"))
        storage_url = f"sqlite:///{db_path}"

    study = optuna.create_study(study_name=args.study_name, storage=storage_url, direction='maximize', load_if_exists=True)
    
    if len(study.trials) == 0:
        study.enqueue_trial({
            'learning_rate': 0.0001,
            'batch_size_sasrec': 512,
            'batch_size_kge': 512,
            'num_neg_b': 4,
            'margin_loss': 5.19,
            'hidden_dropout': 0.4,
            'attn_dropout': 0.3,
        })

    # ensure CSV summary files exist
    trials_csv = os.path.join(study_dir, 'trials_summary.csv')
    best_valid_csv = os.path.join(study_dir, 'best_valid.csv')
    best_test_csv = os.path.join(study_dir, 'best_test.csv')
    if not os.path.exists(trials_csv):
        with open(trials_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'trial', 'timestamp', 'objective_value',
                'valid_recall@20', 'valid_ndcg@20',
                'test_recall@20', 'test_ndcg@20',
                'seed', 'learning_rate', 'batch_size_sasrec', 'batch_size_kge',
                'num_neg_b', 'margin_loss', 'hidden_dropout', 'attn_dropout', 'metrics', 'trial_dir'
            ])
    if not os.path.exists(best_valid_csv):
        with open(best_valid_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'trial', 'best_epoch_valid',
                'learning_rate', 'batch_size_sasrec', 'batch_size_kge', 'num_neg_b',
                'margin_loss', 'hidden_dropout', 'attn_dropout',
                'valid_recall@20', 'valid_ndcg@20',
            ])
    if not os.path.exists(best_test_csv):
        with open(best_test_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'trial', 'best_epoch_test',
                'learning_rate', 'batch_size_sasrec', 'batch_size_kge', 'num_neg_b',
                'margin_loss', 'hidden_dropout', 'attn_dropout',
                'test_recall@20', 'test_ndcg@20',
            ])

    # pass seeds_list into objective via args
    args._seeds_list = seeds_list
    objective = objective_factory(base_cfg, args, study_dir)

    # Optimize (Optuna will persist trials to storage so the job can be resumed)
    study.optimize(objective, n_trials=args.trials)

    print('HPO finished. Best trial:')
    try:
        print(study.best_trial)
    except ValueError:
        print('No completed trials yet (all trials pruned or failed).')


if __name__ == '__main__':
    main()

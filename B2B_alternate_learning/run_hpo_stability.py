import os
import sys
import argparse
import yaml
import json
import csv
import shutil
import subprocess
import time
from pathlib import Path
import statistics


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def write_yaml(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(obj, f)


def run_step4_with_config(cfg_path, epochs=None, env=None, timeout=None):
    cmd = [sys.executable, str(Path(__file__).resolve().parent / "run_step4.py"), '--config', cfg_path]
    if epochs is not None:
        cmd += ['--epochs', str(epochs)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, timeout=timeout)
    return proc.returncode, proc.stdout


def parse_metrics_from_stdout(stdout_text):
    # returns dict with possible keys: val_r20, val_ndcg20, test_r20, test_ndcg20
    res = {'val_r20': None, 'val_ndcg20': None, 'test_r20': None, 'test_ndcg20': None}
    for line in stdout_text.splitlines():
        l = line.lower()
        if '[validation]' in l and 'r@20=' in l and 'ndcg@20=' in l:
            try:
                parts = l.split('r@20=')[-1]
                res['val_r20'] = float(parts.split()[0])
                res['val_ndcg20'] = float(l.split('ndcg@20=')[-1].strip())
            except Exception:
                pass
        if '[test]' in l and 'r@20=' in l and 'ndcg@20=' in l:
            try:
                parts = l.split('r@20=')[-1]
                res['test_r20'] = float(parts.split()[0])
                res['test_ndcg20'] = float(l.split('ndcg@20=')[-1].strip())
            except Exception:
                pass
        # final prints in run_step4: lines like "  R@20:    0.1234"
        if 'r@20:' in l and 'ndcg@20' in l:
            try:
                # attempt to parse both on same line
                parts = l.split('r@20:')[-1]
                r = float(parts.split()[0])
                if 'ndcg@20' in l:
                    nd = float(l.split('ndcg@20:')[-1].split()[0])
                    res['test_r20'] = r
                    res['test_ndcg20'] = nd
            except Exception:
                pass

    return res


def stability_for_config(trial_cfg_path, seeds, epochs, timeout, out_dir, scheduling):
    os.makedirs(out_dir, exist_ok=True)
    records = []
    for seed in seeds:
        # create temporary config with recbole seed override
        cfg = load_yaml(trial_cfg_path)
        recbole_cfg = cfg.get('recbole_config', {})
        recbole_cfg['seed'] = int(seed)
        cfg['recbole_config'] = recbole_cfg
        tmp_cfg_path = os.path.join(out_dir, f'tmp_config_seed_{seed}.yaml')
        write_yaml(cfg, tmp_cfg_path)

        env = os.environ.copy()
        env['PYTHONHASHSEED'] = str(seed)
        env['HPO_TRIAL'] = 'stability'

        ret, out = run_step4_with_config(tmp_cfg_path, epochs=epochs, env=env, timeout=timeout)

        metrics = parse_metrics_from_stdout(out)
        record = {'seed': seed, 'returncode': ret}
        record.update(metrics)
        # save stdout for debugging
        with open(os.path.join(out_dir, f'stdout_seed_{seed}.txt'), 'w', encoding='utf-8') as f:
            f.write(out)
        records.append(record)

    # compute aggregates
    agg = {}
    for key, label in [('test_r20', 'test_r20'), ('test_ndcg20', 'test_ndcg20'), ('val_r20', 'val_r20'), ('val_ndcg20', 'val_ndcg20')]:
        vals = [r[key] for r in records if r.get(key) is not None]
        if vals:
            agg[f'{label}_mean'] = statistics.mean(vals)
            agg[f'{label}_std'] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        else:
            agg[f'{label}_mean'] = None
            agg[f'{label}_std'] = None

    return records, agg


def main():
    parser = argparse.ArgumentParser(description='Post-HPO stability runs for best configs')
    parser.add_argument('--study-dir', type=str, required=True, help='HPO study dir containing best_valid.csv and best_test.csv')
    parser.add_argument('--which', type=str, choices=['valid', 'test', 'both'], default='both')
    parser.add_argument('--seeds', type=str, default=None, help='comma list of seeds to run for stability (default 2024,199,42,1024,888)')
    parser.add_argument('--epochs', type=int, default=None, help='epochs override for stability runs')
    parser.add_argument('--timeout', type=int, default=None, help='timeout seconds for each stability run')
    parser.add_argument('--topk', type=int, default=1, help='how many top configs to test (per file)')
    args = parser.parse_args()

    default_seeds = [2024, 199, 42, 1024, 888]
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    else:
        seeds = default_seeds

    study_dir = args.study_dir
    if not os.path.exists(study_dir):
        print('study-dir not found:', study_dir)
        sys.exit(1)

    out_root = os.path.join(study_dir, 'stability')
    os.makedirs(out_root, exist_ok=True)

    results_summary = {}

    if args.which in ('valid', 'both'):
        bv = os.path.join(study_dir, 'best_valid.csv')
        if os.path.exists(bv):
            with open(bv, 'r', encoding='utf-8') as f:
                reader = list(csv.reader(f))
            # skip header
            rows = reader[1:1+args.topk]
            for i, row in enumerate(rows):
                trial = row[0]
                trial_dir = row[-1] if len(row) > 0 else None
                # trial_dir may be empty; fallback to study_dir/trial_{trial}
                if not trial_dir:
                    trial_dir = os.path.join(study_dir, f'trial_{trial}')
                cfg_path = os.path.join(trial_dir, f'config_trial_{trial}.yaml')
                if not os.path.exists(cfg_path):
                    print('Config not found for trial', trial, 'expected at', cfg_path)
                    continue
                out_dir = os.path.join(out_root, f'best_valid_trial_{trial}')
                print('Running stability for best_valid trial', trial, '->', out_dir)
                records, agg = stability_for_config(cfg_path, seeds, args.epochs, args.timeout, out_dir, scheduling=None)
                results_summary[f'valid_trial_{trial}'] = {'records': records, 'agg': agg}

    if args.which in ('test', 'both'):
        bt = os.path.join(study_dir, 'best_test.csv')
        if os.path.exists(bt):
            with open(bt, 'r', encoding='utf-8') as f:
                reader = list(csv.reader(f))
            rows = reader[1:1+args.topk]
            for i, row in enumerate(rows):
                trial = row[0]
                trial_dir = row[-1] if len(row) > 0 else None
                if not trial_dir:
                    trial_dir = os.path.join(study_dir, f'trial_{trial}')
                cfg_path = os.path.join(trial_dir, f'config_trial_{trial}.yaml')
                if not os.path.exists(cfg_path):
                    print('Config not found for trial', trial, 'expected at', cfg_path)
                    continue
                out_dir = os.path.join(out_root, f'best_test_trial_{trial}')
                print('Running stability for best_test trial', trial, '->', out_dir)
                records, agg = stability_for_config(cfg_path, seeds, args.epochs, args.timeout, out_dir, scheduling=None)
                results_summary[f'test_trial_{trial}'] = {'records': records, 'agg': agg}

    # Save summary
    with open(os.path.join(out_root, 'stability_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2)

    # Also print summary
    print('Stability runs complete. Summary saved to', os.path.join(out_root, 'stability_summary.json'))


if __name__ == '__main__':
    main()

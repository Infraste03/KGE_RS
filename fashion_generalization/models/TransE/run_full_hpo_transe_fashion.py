"""
KGE Fashion - HPO: TransE on the FULL KG (v2, 5 relations).

Use the enriched graph in data/processed_v2/, which includes:
  - compatible_with (element -> element)
  - belongs_to (element -> category)
  - belongs_to_the_brand (item -> brand)
  - belongs_to_price_level (item -> most expensive) [NEW]
  - belongs_to_pop_level (element -> pop_level) [NEW]

It does NOT touch any of the above results:
  - results/hpo_transe/ (v1, 3 relations) INTACT
  - results/hpo_transe_v2/ (search space v2, 3 rel.)  INTACT
  - results/hpo_transe_no_brand/ (unbranded ablation) INTACT

Output NEW to: results/hpo_transe_full/
Studio Optuna NEW: 'transe_fashion_full
"""

import os
import json
import logging
import time
import torch
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths — POINT TO processed_v2, NEW, do not affect v1 ─────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
KG_PATH     = os.path.join(BASE, "data", "processed_v2", "kg_train.tsv")
TASKA_VALID = os.path.join(BASE, "data", "processed_v2", "taskA_valid.tsv")
TASKA_TEST  = os.path.join(BASE, "data", "processed_v2", "taskA_test.tsv")

# ── NEW output — directory never used before ─────────────────────────────────
RESULTS_DIR = os.path.join(BASE, "results", "hpo_transe_full")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── HPO configuration — identical to previous versions for a fair comparison ─
N_TRIALS              = 30
TRAINING_EPOCHS_HPO   = 100
TRAINING_EPOCHS_FINAL = 300
RANDOM_SEED           = 42
HITS_AT_K             = [1, 3, 10]
HPO_TIMEOUT_SECONDS   = 20 * 3600


_file_handler = None

def attach_file_logger(log_path):
    global _file_handler
    root = logging.getLogger()
    if _file_handler is not None:
        root.removeHandler(_file_handler)
        _file_handler.close()
    _file_handler = logging.FileHandler(log_path, mode='a')
    _file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    root.addHandler(_file_handler)


def load_triples_factories():
    logger.info("Loading triples (full KG, 5 relations)...")
    train_df = pd.read_csv(KG_PATH,     sep='\t')
    valid_df = pd.read_csv(TASKA_VALID, sep='\t')
    test_df  = pd.read_csv(TASKA_TEST,  sep='\t')

    logger.info(f"  Train: {len(train_df):,} triple")
    logger.info(f"  Valid: {len(valid_df):,} triple")
    logger.info(f"  Test:  {len(test_df):,} triple")
    logger.info(f"  Relations in train: "
                f"{sorted(train_df['relation'].unique().tolist())}")

    train_arr = train_df[['head', 'relation', 'tail']].values
    valid_arr = valid_df[['head', 'relation', 'tail']].values
    test_arr  = test_df[['head',  'relation', 'tail']].values

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
    logger.info(f"  Entities:   {train_factory.num_entities:,}")
    logger.info(f"  Relations:{train_factory.num_relations:,}")

    if 'compatible_with' not in train_factory.relation_to_id:
        raise ValueError("Relation 'compatible_with' not found in the KG")

    return train_factory, valid_factory, test_factory


def build_type_index(train_factory):
    """
    Full KG: 5 entity types.
    item, category, brand, pricetier, poptier.
    """
    logger.info("=" * 70)
    logger.info("BUILDING TYPE INDEX (5 types)")
    logger.info("=" * 70)

    type_to_ids = {
        'item': [], 'category': [], 'brand': [],
        'pricetier': [], 'poptier': [],
    }
    unknown = []

    for label, eid in train_factory.entity_to_id.items():
        prefix = label.split('_')[0]
        if prefix in type_to_ids:
            type_to_ids[prefix].append(eid)
        else:
            unknown.append(label)

    if unknown:
        logger.warning(f"  {len(unknown):,} entities with unknown prefix")
        logger.warning(f"  Example: {unknown[:3]}")

    for k, v in type_to_ids.items():
        logger.info(f"  type '{k}': {len(v):,} entities")

    return {k: torch.LongTensor(v) for k, v in type_to_ids.items()}


def suggest_hyperparameters(trial):
    return {
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


def build_pipeline_kwargs(params, num_epochs, device, use_early_stopping=False):
    kwargs = dict(
        model='TransE',
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
    if use_early_stopping:
        kwargs['stopper'] = 'early'
        kwargs['stopper_kwargs'] = {
            'metric': 'mrr',
            'patience': 10,
            'relative_delta': 0.002,
            'larger_is_better': True,
        }
    return kwargs


def make_objective(train_factory, valid_factory, device):
    def objective(trial):
        params = suggest_hyperparameters(trial)
        logger.info(f"  Trial {trial.number}: {params}")
        try:
            result = pipeline(
                training=train_factory,
                validation=valid_factory,
                testing=valid_factory,
                **build_pipeline_kwargs(params, TRAINING_EPOCHS_HPO, device)
            )
            mrr = result.metric_results.get_metric('mrr')
            logger.info(f"  Trial {trial.number} -> MRR={mrr:.4f}")
            return mrr
        except Exception as e:
            logger.warning(f"  Trial {trial.number} failed: {e}")
            return 0.0
    return objective


def compute_metrics_from_ranks(ranks, hits_at_k):
    ranks = ranks.float()
    metrics = {
        'MRR': (1.0 / ranks).mean().item(),
        'MR':  ranks.mean().item(),
    }
    for k in hits_at_k:
        metrics[f'Hits@{k}'] = (ranks <= k).float().mean().item()
    return metrics


def build_known_targets_index(factories, relation_id):
    head_to_tails = {}
    tail_to_heads = {}
    for fac in factories:
        for h, r, t in fac.mapped_triples.tolist():
            if r != relation_id:
                continue
            head_to_tails.setdefault(h, set()).add(t)
            tail_to_heads.setdefault(t, set()).add(h)
    return head_to_tails, tail_to_heads


def evaluate_custom(model, test_factory, train_factory,
                    valid_factory, type_index, mode):
    logger.info(f"  Evaluation mode: {mode}")
    device       = next(model.parameters()).device
    relation_id  = train_factory.relation_to_id['compatible_with']
    num_entities = train_factory.num_entities

    if mode == 'type_constrained':
        # compatible_with is always item -> item, also in the enriched KG
        item_ids  = type_index['item']
        tail_mask = torch.full((num_entities,), float('-inf'))
        tail_mask[item_ids] = 0.0
        head_mask = torch.full((num_entities,), float('-inf'))
        head_mask[item_ids] = 0.0
    else:
        tail_mask = torch.zeros(num_entities)
        head_mask = torch.zeros(num_entities)

    tail_mask = tail_mask.to(device)
    head_mask = head_mask.to(device)

    head_to_tails, tail_to_heads = build_known_targets_index(
        [train_factory, valid_factory, test_factory], relation_id
    )

    test_triples = test_factory.mapped_triples
    test_triples = test_triples[test_triples[:, 1] == relation_id]
    n_test = len(test_triples)
    logger.info(f"    test triples: {n_test:,}")

    model.eval()
    tail_ranks = []
    head_ranks = []

    with torch.no_grad():
        for h, r, t in test_triples.tolist():
            hr     = torch.tensor([[h, r]], device=device)
            scores = model.score_t(hr).squeeze(0) + tail_mask
            for kt in head_to_tails.get(h, set()):
                if kt != t:
                    scores[kt] = float('-inf')
            tail_ranks.append((scores > scores[t].item()).sum().item() + 1)

            rt     = torch.tensor([[r, t]], device=device)
            scores = model.score_h(rt).squeeze(0) + head_mask
            for kh in tail_to_heads.get(t, set()):
                if kh != h:
                    scores[kh] = float('-inf')
            head_ranks.append((scores > scores[h].item()).sum().item() + 1)

    tail_m = compute_metrics_from_ranks(torch.tensor(tail_ranks), HITS_AT_K)
    head_m = compute_metrics_from_ranks(torch.tensor(head_ranks), HITS_AT_K)
    avg_m  = {k: (tail_m[k] + head_m[k]) / 2 for k in tail_m}

    for label, m in [('tail', tail_m), ('head', head_m), ('avg', avg_m)]:
        logger.info(f"    --- {label} ---")
        for k, v in m.items():
            logger.info(f"      {k:8s}: {v:.4f}")

    return {'tail': tail_m, 'head': head_m, 'avg': avg_m}


def retrain_and_evaluate_best(best_params, train_factory,
                               valid_factory, test_factory,
                               type_index, device):
    logger.info(f"  Final retraining with best params, "
                f"up to {TRAINING_EPOCHS_FINAL} epochs (early stopping)...")
    result = pipeline(
        training=train_factory,
        validation=valid_factory,
        testing=valid_factory,
        **build_pipeline_kwargs(
            best_params, TRAINING_EPOCHS_FINAL, device,
            use_early_stopping=True
        )
    )
    naive = evaluate_custom(result.model, test_factory,
                            train_factory, valid_factory,
                            type_index, 'naive')
    typed = evaluate_custom(result.model, test_factory,
                            train_factory, valid_factory,
                            type_index, 'type_constrained')
    return result.model, {'naive': naive, 'type_constrained': typed}


def main():
    attach_file_logger(os.path.join(RESULTS_DIR, 'run.log'))

    logger.info("=" * 70)
    logger.info("FASHION — TRANSE HPO — FULL KG (5 relations)")
    logger.info("=" * 70)
    logger.info(f"  Data:           {os.path.dirname(KG_PATH)}")
    logger.info(f"  Trials:         {N_TRIALS}")
    logger.info(f"  HPO epochs:     {TRAINING_EPOCHS_HPO}")
    logger.info(f"  Final epochs:   {TRAINING_EPOCHS_FINAL}")
    logger.info(f"  Output:         {RESULTS_DIR}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"  Device:         {device}")
    if device == 'cuda':
        logger.info(f"  GPU:            {torch.cuda.get_device_name(0)}")

    train_factory, valid_factory, test_factory = load_triples_factories()
    type_index = build_type_index(train_factory)

    # ── NEW Optuna study — does not conflict with the others ─────────────────
    storage_path = os.path.join(RESULTS_DIR, 'study.db')
    storage_url  = f'sqlite:///{os.path.abspath(storage_path)}'

    study = optuna.create_study(
        study_name='transe_fashion_full',   # new name, never used before
        storage=storage_url,
        direction='maximize',
        sampler=TPESampler(seed=RANDOM_SEED),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        load_if_exists=True,
    )

    n_done      = sum(1 for t in study.trials
                      if t.state == optuna.trial.TrialState.COMPLETE)
    n_remaining = max(0, N_TRIALS - n_done)
    logger.info(f"  Completed trials: {n_done}")
    logger.info(f"  Remaining trials:  {n_remaining}")

    if n_remaining > 0:
        objective = make_objective(train_factory, valid_factory, device)
        study.optimize(
            objective,
            n_trials=n_remaining,
            timeout=HPO_TIMEOUT_SECONDS,
            gc_after_trial=True,
        )

    n_done_after = sum(1 for t in study.trials
                       if t.state == optuna.trial.TrialState.COMPLETE)
    logger.info(f"  HPO completed. Total trials: {n_done_after}")

    if n_done_after == 0:
        logger.error("  No trial completed. Exiting.")
        return

    logger.info(f"  Best MRR (validation): {study.best_value:.4f}")
    logger.info(f"  Best params: {study.best_params}")

    with open(os.path.join(RESULTS_DIR, 'best_params.json'), 'w') as f:
        json.dump(study.best_params, f, indent=2)
    study.trials_dataframe().to_csv(
        os.path.join(RESULTS_DIR, 'trials.csv'), index=False
    )

    final_model, test_metrics = retrain_and_evaluate_best(
        study.best_params, train_factory, valid_factory,
        test_factory, type_index, device
    )

    torch.save(final_model.state_dict(),
               os.path.join(RESULTS_DIR, 'best_model.pt'))
    with open(os.path.join(RESULTS_DIR, 'best_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS — FULL KG (TEST set, avg head+tail)")
    logger.info("=" * 70)
    for mode in ['naive', 'type_constrained']:
        avg = test_metrics[mode]['avg']
        logger.info(
            f"  {mode:20s} | MRR={avg['MRR']:.4f} "
            f"H@1={avg['Hits@1']:.4f} "
            f"H@3={avg['Hits@3']:.4f} "
            f"H@10={avg['Hits@10']:.4f}"
        )

    # ── Comparison with v1 (3 relations) and no-brand, if available ──────────
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON WITH OTHER VERSIONS")
    logger.info("=" * 70)

    comparisons = {
        'v1 (3 relations, with brand)': os.path.join(
            BASE, "results", "hpo_transe", "best_metrics.json"),
        'no_brand (2 relations)': os.path.join(
            BASE, "results", "hpo_transe_no_brand", "best_metrics.json"),
    }

    for label, path in comparisons.items():
        if os.path.exists(path):
            with open(path) as f:
                other_metrics = json.load(f)
            for mode in ['naive', 'type_constrained']:
                other_mrr = other_metrics[mode]['avg']['MRR']
                new_mrr   = test_metrics[mode]['avg']['MRR']
                delta     = new_mrr - other_mrr
                pct       = (delta / other_mrr * 100) if other_mrr > 0 else 0
                logger.info(
                    f"  [{mode}] full vs {label}: "
                    f"{other_mrr:.4f} -> {new_mrr:.4f} "
                    f"(delta={delta:+.4f}, {pct:+.1f}%)"
                )
        else:
            logger.warning(f"  File not found for comparison '{label}': {path}")

    logger.info("\nDONE.")


if __name__ == "__main__":
    main()
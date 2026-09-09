"""
Fashion KGE — HPO: TransE with ALL 5 relations (enriched KG, processed_v2).

Unlike the ablation variants (no_brand/no_category/no_compatible_with),
this is NOT a filtered version of the original graph: it is a DIFFERENT graph,
with 2 additional relations (belongs_to_price_tier, belongs_to_pop_tier) and
more entities (182,990 compared to 182,983 in the 3-relation graph — 7 additional
entities corresponding to the shared price/popularity tiers).

No fixed vocabulary needs to be derived here: no relations are filtered from
the complete graph, and the entire kg_train_v2.tsv is used as is.
The PyKEEN automatically inferred vocabulary is therefore appropriate and
deterministic.

ONLY correction compared with the old run_full_hpo_transe_fashion.py:
embedding_dim is FIXED to 64 (previously it was part of the search space and
128 was selected, which is incompatible with kge_dim=64 used in Step 4
for warm-starting).

Output in results/hpo_transe_full5rel_v2/ (separate from the historical
hpo_transe_full5rel/ results with embedding_dim=128).
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

# ── Paths — processed_v2, NOT processed ──────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

KG_PATH     = os.path.join(BASE, "data", "processed_v2", "kg_train.tsv")
TASKA_VALID = os.path.join(BASE, "data", "processed_v2", "taskA_valid.tsv")
TASKA_TEST  = os.path.join(BASE, "data", "processed_v2", "taskA_test.tsv")

RESULTS_DIR = os.path.join(BASE, "results", "hpo_transe_full5rel_v2")
os.makedirs(RESULTS_DIR, exist_ok=True)

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
    logger.info("Loading triples — FULL KG with 5 relations (processed_v2)")

    train_df = pd.read_csv(KG_PATH,     sep='\t')
    valid_df = pd.read_csv(TASKA_VALID, sep='\t')
    test_df  = pd.read_csv(TASKA_TEST,  sep='\t')

    logger.info(f"  Relations: {sorted(train_df['relation'].unique().tolist())}")

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
    logger.info(f"  Relations:  {train_factory.num_relations:,}")
    logger.info(f"  Train:      {train_factory.num_triples:,} triples")
    logger.info(f"  Valid:      {valid_factory.num_triples:,} triples")
    logger.info(f"  Test:       {test_factory.num_triples:,} triples")
    return train_factory, valid_factory, test_factory


def build_type_index(train_factory):
    # Prefixes also include pricetier/poptier, not only item/brand/category
    type_to_ids = {}
    for label, eid in train_factory.entity_to_id.items():
        prefix = label.split('_')[0]
        type_to_ids.setdefault(prefix, []).append(eid)
    for k, v in type_to_ids.items():
        logger.info(f"  type '{k}': {len(v):,} entities")
    return {k: torch.LongTensor(v) for k, v in type_to_ids.items()}


def suggest_hyperparameters(trial):
    return {
        'embedding_dim': 64,   # FIXED — must match kge_dim used in Step 4
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
    logger.info("FASHION — HPO TransE — FULL 5-RELATION KG (fixed dim=64)")
    logger.info("=" * 70)
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

    storage_path = os.path.join(RESULTS_DIR, 'study.db')
    storage_url  = f'sqlite:///{os.path.abspath(storage_path)}'

    study = optuna.create_study(
        study_name='transe_fashion_full5rel_v2',
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
    logger.info(f"  Best params (Optuna): {study.best_params}")

    # FIX: embedding_dim is fixed (not handled by Optuna) and must be added manually
    best_params = dict(study.best_params)
    best_params['embedding_dim'] = 64
    logger.info(f"  Best params (complete): {best_params}")

    with open(os.path.join(RESULTS_DIR, 'best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=2)
    study.trials_dataframe().to_csv(
        os.path.join(RESULTS_DIR, 'trials.csv'), index=False
    )

    final_model, test_metrics = retrain_and_evaluate_best(
        best_params, train_factory, valid_factory,
        test_factory, type_index, device
    )

    torch.save(final_model.state_dict(),
               os.path.join(RESULTS_DIR, 'best_model.pt'))
    with open(os.path.join(RESULTS_DIR, 'best_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS — TransE 5 RELATIONS, dim=64 (TEST set, avg head+tail)")
    logger.info("=" * 70)
    for mode in ['naive', 'type_constrained']:
        avg = test_metrics[mode]['avg']
        logger.info(
            f"  {mode:20s} | MRR={avg['MRR']:.4f} "
            f"H@1={avg['Hits@1']:.4f} "
            f"H@3={avg['Hits@3']:.4f} "
            f"H@10={avg['Hits@10']:.4f}"
        )

    logger.info("\nDONE.")


if __name__ == "__main__":
    main()
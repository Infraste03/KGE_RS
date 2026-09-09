"""
Step 2: Train and evaluate KGE models on Task A (compatible_with link prediction).

Trains three models (TransE, RotatE, ComplEx) on the same KG and evaluates
each in two modes:

  - naive            : rank against ALL entities (standard KGE benchmark)
  - type_constrained : rank only against entities of the correct type

Metrics reported (filtered, averaged head+tail):
  MRR, Hits@1, Hits@3, Hits@10, Mean Rank
"""

import os
import json
import logging
import time
import torch
import numpy as np
import pandas as pd

from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

KG_PATH = os.path.join("data", "processed", "kg_train.tsv")
TASKA_VALID = os.path.join("data", "processed", "taskA_valid.tsv")
TASKA_TEST = os.path.join("data", "processed", "taskA_test.tsv")
RESULTS_DIR = os.path.join("results", "step2_kge")

MODELS_TO_TRAIN = {
    'TransE': 200,
    'RotatE': 200,
    'ComplEx': 200,
    'DistMult': 200,
}

NUM_EPOCHS = 100
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
NUM_NEGS_PER_POS = 32
RANDOM_SEED = 42

HITS_AT_K = [1, 3, 10, 20]


# --------------------------------------------------------------------------- #
# GPU diagnostics
# --------------------------------------------------------------------------- #

def diagnose_gpu():
    """Check GPU availability and report diagnostic info."""
    logger.info("=" * 70)
    logger.info("GPU DIAGNOSTICS")
    logger.info("=" * 70)

    cuda_available = torch.cuda.is_available()
    logger.info(f"  PyTorch version       : {torch.__version__}")
    logger.info(f"  CUDA available        : {cuda_available}")

    if cuda_available:
        n_devices = torch.cuda.device_count()
        logger.info(f"  CUDA devices          : {n_devices}")
        for i in range(n_devices):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / (1024 ** 3)
            logger.info(f"    Device {i}: {name} ({mem_gb:.1f} GB)")
        logger.info(f"  CUDA version (PyTorch): {torch.version.cuda}")
        device = 'cuda'
    else:
        logger.warning("  !!! CUDA is NOT available -- training will run on CPU.")
        logger.warning("  !!! This will be SLOW for RotatE/ComplEx on this dataset.")
        logger.warning("  !!! To enable GPU, install PyTorch with CUDA support:")
        logger.warning("  !!!   pip install torch --index-url "
                       "https://download.pytorch.org/whl/cu121")
        device = 'cpu'

    logger.info(f"  Selected device       : {device}")
    return device


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_triples_factories():
    """Load KG, valid, test triples with shared entity/relation namespaces."""
    logger.info("=" * 70)
    logger.info("LOADING TRIPLES")
    logger.info("=" * 70)

    for path in [KG_PATH, TASKA_VALID, TASKA_TEST]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")
        logger.info(f"  Found: {path}")

    train_df = pd.read_csv(KG_PATH, sep='\t')
    valid_df = pd.read_csv(TASKA_VALID, sep='\t')
    test_df = pd.read_csv(TASKA_TEST, sep='\t')

    logger.info(f"  Train rows : {len(train_df):,}")
    logger.info(f"  Valid rows : {len(valid_df):,}")
    logger.info(f"  Test rows  : {len(test_df):,}")

    train_arr = train_df[['head', 'relation', 'tail']].values
    valid_arr = valid_df[['head', 'relation', 'tail']].values
    test_arr = test_df[['head', 'relation', 'tail']].values

    logger.info("  Building TriplesFactory for training...")
    train_factory = TriplesFactory.from_labeled_triples(train_arr)
    logger.info(f"    Entities : {train_factory.num_entities:,}")
    logger.info(f"    Relations: {train_factory.num_relations:,}")
    logger.info(f"    Triples  : {train_factory.num_triples:,}")

    logger.info("  Building TriplesFactory for valid (sharing namespace)...")
    valid_factory = TriplesFactory.from_labeled_triples(
        valid_arr,
        entity_to_id=train_factory.entity_to_id,
        relation_to_id=train_factory.relation_to_id,
    )
    logger.info(f"    Triples  : {valid_factory.num_triples:,}")

    logger.info("  Building TriplesFactory for test (sharing namespace)...")
    test_factory = TriplesFactory.from_labeled_triples(
        test_arr,
        entity_to_id=train_factory.entity_to_id,
        relation_to_id=train_factory.relation_to_id,
    )
    logger.info(f"    Triples  : {test_factory.num_triples:,}")

    if 'compatible_with' not in train_factory.relation_to_id:
        raise ValueError("'compatible_with' relation not found in training KG")
    logger.info("  Sanity check: 'compatible_with' relation found.")

    return train_factory, valid_factory, test_factory


def build_type_index(train_factory):
    """Group entity IDs by type prefix (item / machine / client / model)."""
    logger.info("=" * 70)
    logger.info("BUILDING TYPE INDEX")
    logger.info("=" * 70)

    type_to_ids = {'item': [], 'machine': [], 'client': [], 'model': []}
    unknown = []

    for label, eid in train_factory.entity_to_id.items():
        prefix = label.split('_')[0]
        if prefix in type_to_ids:
            type_to_ids[prefix].append(eid)
        else:
            unknown.append(label)

    if unknown:
        logger.warning(f"  Found {len(unknown):,} entities with unknown prefix")
        logger.warning(f"    sample: {unknown[:5]}")

    for k, v in type_to_ids.items():
        logger.info(f"  type '{k}': {len(v):,} entities")

    total = sum(len(v) for v in type_to_ids.values()) + len(unknown)
    logger.info(f"  total: {total:,} (must equal {train_factory.num_entities:,})")

    if total != train_factory.num_entities:
        raise RuntimeError(
            f"Type index covers {total} entities but factory has "
            f"{train_factory.num_entities}"
        )

    return {k: torch.LongTensor(v) for k, v in type_to_ids.items()}


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train_model(model_name, embedding_dim, train_factory, valid_factory, device):
    """Train a single KGE model using PyKEEN's pipeline."""
    logger.info("=" * 70)
    logger.info(f"TRAINING {model_name} (dim={embedding_dim}, epochs={NUM_EPOCHS})")
    logger.info("=" * 70)
    logger.info(f"  optimizer       : Adam (lr={LEARNING_RATE})")
    logger.info(f"  batch size      : {BATCH_SIZE}")
    logger.info(f"  negatives/pos   : {NUM_NEGS_PER_POS}")
    logger.info(f"  device          : {device}")
    logger.info(f"  random seed     : {RANDOM_SEED}")

    t0 = time.time()
    result = pipeline(
        training=train_factory,
        validation=valid_factory,
        testing=valid_factory,
        model=model_name,
        model_kwargs={'embedding_dim': embedding_dim},
        optimizer='Adam',
        optimizer_kwargs={'lr': LEARNING_RATE},
        training_kwargs={
            'num_epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
        },
        negative_sampler='basic',
        negative_sampler_kwargs={'num_negs_per_pos': NUM_NEGS_PER_POS},
        random_seed=RANDOM_SEED,
        device=device,
    )
    elapsed = time.time() - t0
    logger.info(f"  Training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return result.model


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def compute_metrics_from_ranks(ranks, hits_at_k):
    """Given a 1-D tensor of ranks (1-indexed), compute MRR, MR, Hits@k."""
    ranks = ranks.float()
    metrics = {
        'MRR': (1.0 / ranks).mean().item(),
        'MR': ranks.mean().item(),
    }
    for k in hits_at_k:
        metrics[f'Hits@{k}'] = (ranks <= k).float().mean().item()
    return metrics


def build_known_targets_index(factories, relation_id):
    """Build dictionaries of known (h,r,t) for filtered evaluation."""
    logger.info("    Building filtered-evaluation index...")
    head_to_tails = {}
    tail_to_heads = {}
    total = 0
    for fac in factories:
        triples = fac.mapped_triples
        for h, r, t in triples.tolist():
            if r != relation_id:
                continue
            head_to_tails.setdefault(h, set()).add(t)
            tail_to_heads.setdefault(t, set()).add(h)
            total += 1
    logger.info(f"    Indexed {total:,} known compatible_with triples "
                f"(across train+valid+test)")
    return head_to_tails, tail_to_heads


def evaluate(
    model, test_factory, train_factory, valid_factory, type_index, mode,
    relation_label='compatible_with',
):
    """Evaluate model on test triples in 'naive' or 'type_constrained' mode."""
    logger.info("-" * 70)
    logger.info(f"  EVALUATION mode: {mode}")
    logger.info("-" * 70)

    device = next(model.parameters()).device
    relation_id = train_factory.relation_to_id[relation_label]

    num_entities = train_factory.num_entities
    if mode == 'type_constrained':
        tail_mask = torch.full((num_entities,), float('-inf'))
        tail_mask[type_index['machine']] = 0.0
        head_mask = torch.full((num_entities,), float('-inf'))
        head_mask[type_index['item']] = 0.0
        logger.info(f"    tail candidates (machines): {len(type_index['machine']):,}")
        logger.info(f"    head candidates (items)   : {len(type_index['item']):,}")
    else:
        tail_mask = torch.zeros(num_entities)
        head_mask = torch.zeros(num_entities)
        logger.info(f"    candidates (all entities) : {num_entities:,}")

    tail_mask = tail_mask.to(device)
    head_mask = head_mask.to(device)

    head_to_known_tails, tail_to_known_heads = build_known_targets_index(
        [train_factory, valid_factory, test_factory],
        relation_id,
    )

    test_triples = test_factory.mapped_triples
    rel_mask = test_triples[:, 1] == relation_id
    test_triples = test_triples[rel_mask]
    n_test = len(test_triples)
    logger.info(f"    test triples to evaluate  : {n_test:,}")

    model.eval()
    tail_ranks = []
    head_ranks = []

    log_every = max(1, n_test // 5)
    t0 = time.time()

    with torch.no_grad():
        for i, (h, r, t) in enumerate(test_triples.tolist()):
            # ---- tail prediction ----
            hr = torch.tensor([[h, r]], device=device)
            scores = model.score_t(hr).squeeze(0)
            scores = scores + tail_mask
            for kt in head_to_known_tails.get(h, set()):
                if kt != t:
                    scores[kt] = float('-inf')
            true_score = scores[t].item()
            rank = (scores > true_score).sum().item() + 1
            tail_ranks.append(rank)

            # ---- head prediction ----
            rt = torch.tensor([[r, t]], device=device)
            scores = model.score_h(rt).squeeze(0)
            scores = scores + head_mask
            for kh in tail_to_known_heads.get(t, set()):
                if kh != h:
                    scores[kh] = float('-inf')
            true_score = scores[h].item()
            rank = (scores > true_score).sum().item() + 1
            head_ranks.append(rank)

            if (i + 1) % log_every == 0:
                elapsed = time.time() - t0
                pct = 100 * (i + 1) / n_test
                logger.info(f"    progress: {i+1:,}/{n_test:,} "
                            f"({pct:.0f}%, {elapsed:.1f}s)")

    elapsed = time.time() - t0
    logger.info(f"    evaluation completed in {elapsed:.1f}s")

    tail_ranks = torch.tensor(tail_ranks)
    head_ranks = torch.tensor(head_ranks)

    tail_m = compute_metrics_from_ranks(tail_ranks, HITS_AT_K)
    head_m = compute_metrics_from_ranks(head_ranks, HITS_AT_K)
    avg_m = {k: (tail_m[k] + head_m[k]) / 2 for k in tail_m}

    logger.info("    --- tail prediction ---")
    for k, v in tail_m.items():
        logger.info(f"      {k:8s}: {v:.4f}")
    logger.info("    --- head prediction ---")
    for k, v in head_m.items():
        logger.info(f"      {k:8s}: {v:.4f}")
    logger.info("    --- averaged head+tail ---")
    for k, v in avg_m.items():
        logger.info(f"      {k:8s}: {v:.4f}")

    return {'tail': tail_m, 'head': head_m, 'avg': avg_m}


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    logger.info("=" * 70)
    logger.info("STEP 2: KGE MODELS TRAINING AND EVALUATION")
    logger.info("=" * 70)
    logger.info(f"  Models to train : {list(MODELS_TO_TRAIN.keys())}")
    logger.info(f"  Results dir     : {RESULTS_DIR}")

    device = diagnose_gpu()
    train_factory, valid_factory, test_factory = load_triples_factories()
    type_index = build_type_index(train_factory)

    all_results = {}

    for idx, (model_name, embedding_dim) in enumerate(MODELS_TO_TRAIN.items(), 1):
        logger.info("\n" + "#" * 70)
        logger.info(f"# MODEL {idx}/{len(MODELS_TO_TRAIN)}: {model_name}")
        logger.info("#" * 70)

        model = train_model(
            model_name, embedding_dim, train_factory, valid_factory, device
        )

        model_path = os.path.join(RESULTS_DIR, f'{model_name.lower()}.pt')
        torch.save(model.state_dict(), model_path)
        logger.info(f"  Saved model weights to {model_path}")

        logger.info(f"\n  Running evaluation for {model_name}...")
        results_naive = evaluate(
            model, test_factory, train_factory, valid_factory,
            type_index, mode='naive',
        )
        results_typed = evaluate(
            model, test_factory, train_factory, valid_factory,
            type_index, mode='type_constrained',
        )

        all_results[model_name] = {
            'naive': results_naive,
            'type_constrained': results_typed,
        }

        partial_path = os.path.join(RESULTS_DIR, 'results.json')
        with open(partial_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"  Partial results saved to {partial_path}")

    # ---- Final comparison table ----
    logger.info("\n" + "=" * 70)
    logger.info("FINAL COMPARISON TABLE (averaged head+tail)")
    logger.info("=" * 70)
    header = (f"{'Model':10s} {'Mode':18s} {'MRR':>8s} "
              f"{'H@1':>8s} {'H@3':>8s} {'H@10':>8s} {'MR':>8s}")
    logger.info(header)
    logger.info("-" * len(header))
    for model_name in MODELS_TO_TRAIN:
        for mode in ['naive', 'type_constrained']:
            avg = all_results[model_name][mode]['avg']
            logger.info(
                f"{model_name:10s} {mode:18s} "
                f"{avg['MRR']:>8.4f} {avg['Hits@1']:>8.4f} "
                f"{avg['Hits@3']:>8.4f} {avg['Hits@10']:>8.4f} "
                f"{avg['MR']:>8.1f}"
            )

    logger.info("\nDONE.")


if __name__ == "__main__":
    main()
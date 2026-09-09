"""
Fashion KGE — Step 1: Train TransE on Task A (compatible_with link prediction).

Trains TransE on the Fashion KG and evaluates link prediction
for the compatible_with relation in two modes:
  - naive           : rank against ALL entities
  - type_constrained: rank only against items (both head and tail are items)
"""

import os
import json
import logging
import time
import torch
import pandas as pd

from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

KG_PATH     = os.path.join(BASE, "data", "processed", "kg_train.tsv")
TASKA_VALID = os.path.join(BASE, "data", "processed", "taskA_valid.tsv")
TASKA_TEST  = os.path.join(BASE, "data", "processed", "taskA_test.tsv")
RESULTS_DIR = os.path.join(BASE, "results", "transe")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Hyperparameters (smoke test — reasonable values for local testing) ────────
EMBEDDING_DIM    = 64       # embedding dimension to test
NUM_EPOCHS       = 50       # low for local smoke test, then 100+ on HPC
BATCH_SIZE       = 512
LEARNING_RATE    = 1e-3
NUM_NEGS_PER_POS = 32
RANDOM_SEED      = 42
HITS_AT_K        = [1, 3, 10, 20]


# ── GPU diagnostics ───────────────────────────────────────────────────────────
def diagnose_gpu():
    logger.info("=" * 70)
    logger.info("GPU DIAGNOSTICS")
    logger.info("=" * 70)
    cuda_available = torch.cuda.is_available()
    logger.info(f"  PyTorch version : {torch.__version__}")
    logger.info(f"  CUDA available  : {cuda_available}")
    if cuda_available:
        for i in range(torch.cuda.device_count()):
            name  = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            mem   = props.total_memory / (1024 ** 3)
            logger.info(f"    Device {i}: {name} ({mem:.1f} GB)")
        device = 'cuda'
    else:
        logger.warning("  CUDA not available — training on CPU (slow)")
        device = 'cpu'
    logger.info(f"  Selected device: {device}")
    return device


# ── Triple loading ─────────────────────────────────────────────────────────────
def load_triples_factories():
    logger.info("=" * 70)
    logger.info("LOADING TRIPLES")
    logger.info("=" * 70)

    for path in [KG_PATH, TASKA_VALID, TASKA_TEST]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        logger.info(f"  Found: {path}")

    train_df = pd.read_csv(KG_PATH,     sep='\t')
    valid_df = pd.read_csv(TASKA_VALID, sep='\t')
    test_df  = pd.read_csv(TASKA_TEST,  sep='\t')

    logger.info(f"  Train: {len(train_df):,} triples")
    logger.info(f"  Valid: {len(valid_df):,} triples")
    logger.info(f"  Test:  {len(test_df):,} triples")

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

    logger.info(f"  Entities: {train_factory.num_entities:,}")
    logger.info(f"  Relations: {train_factory.num_relations:,}")

    if 'compatible_with' not in train_factory.relation_to_id:
        raise ValueError("Relation 'compatible_with' not found in the KG")
    logger.info("  Sanity check: 'compatible_with' found.")

    return train_factory, valid_factory, test_factory


# ── Type index ─────────────────────────────────────────────────────────────────
def build_type_index(train_factory):
    """
    In Fashion, all compatible_with nodes are items.
    Groups entities by prefix: item, category, brand.
    """
    logger.info("=" * 70)
    logger.info("BUILDING TYPE INDEX")
    logger.info("=" * 70)

    type_to_ids = {'item': [], 'category': [], 'brand': []}
    unknown = []

    for label, eid in train_factory.entity_to_id.items():
        prefix = label.split('_')[0]
        if prefix in type_to_ids:
            type_to_ids[prefix].append(eid)
        else:
            unknown.append(label)

    if unknown:
        logger.warning(f"  {len(unknown):,} entities with unknown prefix")

    for k, v in type_to_ids.items():
        logger.info(f"  type '{k}': {len(v):,} entities")

    return {k: torch.LongTensor(v) for k, v in type_to_ids.items()}


# ── Training ───────────────────────────────────────────────────────────────────
def train_model(train_factory, valid_factory, device):
    logger.info("=" * 70)
    logger.info(f"TRAINING TransE (dim={EMBEDDING_DIM}, epochs={NUM_EPOCHS})")
    logger.info("=" * 70)

    t0 = time.time()
    result = pipeline(
        training=train_factory,
        validation=valid_factory,
        testing=valid_factory,
        model='TransE',
        model_kwargs={'embedding_dim': EMBEDDING_DIM},
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


# ── Evaluation ─────────────────────────────────────────────────────────────────
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
    logger.info("    Building filtered-evaluation index...")
    head_to_tails = {}
    tail_to_heads = {}
    total = 0
    for fac in factories:
        for h, r, t in fac.mapped_triples.tolist():
            if r != relation_id:
                continue
            head_to_tails.setdefault(h, set()).add(t)
            tail_to_heads.setdefault(t, set()).add(h)
            total += 1
    logger.info(f"    Indexed {total:,} compatible_with triples")
    return head_to_tails, tail_to_heads


def evaluate(model, test_factory, train_factory, valid_factory,
             type_index, mode):
    """
    Evaluates TransE on compatible_with.

    In Fashion, compatible_with is item->item, therefore:
    - naive: rank against all entities
    - type_constrained: rank only against items (both head and tail)
    """
    logger.info("-" * 70)
    logger.info(f"  EVALUATION mode: {mode}")
    logger.info("-" * 70)

    device      = next(model.parameters()).device
    relation_id = train_factory.relation_to_id['compatible_with']
    num_entities = train_factory.num_entities

    if mode == 'type_constrained':
        # Both head and tail are items
        item_ids = type_index['item']
        tail_mask = torch.full((num_entities,), float('-inf'))
        tail_mask[item_ids] = 0.0
        head_mask = torch.full((num_entities,), float('-inf'))
        head_mask[item_ids] = 0.0
        logger.info(f"    item candidates: {len(item_ids):,}")
    else:
        tail_mask = torch.zeros(num_entities)
        head_mask = torch.zeros(num_entities)
        logger.info(f"    candidates (all entities): {num_entities:,}")

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
    log_every  = max(1, n_test // 5)
    t0         = time.time()

    with torch.no_grad():
        for i, (h, r, t) in enumerate(test_triples.tolist()):
            # Tail prediction
            hr     = torch.tensor([[h, r]], device=device)
            scores = model.score_t(hr).squeeze(0) + tail_mask
            for kt in head_to_tails.get(h, set()):
                if kt != t:
                    scores[kt] = float('-inf')
            rank = (scores > scores[t].item()).sum().item() + 1
            tail_ranks.append(rank)

            # Head prediction
            rt     = torch.tensor([[r, t]], device=device)
            scores = model.score_h(rt).squeeze(0) + head_mask
            for kh in tail_to_heads.get(t, set()):
                if kh != h:
                    scores[kh] = float('-inf')
            rank = (scores > scores[h].item()).sum().item() + 1
            head_ranks.append(rank)

            if (i + 1) % log_every == 0:
                logger.info(f"    {i+1:,}/{n_test:,} "
                            f"({100*(i+1)/n_test:.0f}%, "
                            f"{time.time()-t0:.1f}s)")

    tail_m = compute_metrics_from_ranks(torch.tensor(tail_ranks), HITS_AT_K)
    head_m = compute_metrics_from_ranks(torch.tensor(head_ranks), HITS_AT_K)
    avg_m  = {k: (tail_m[k] + head_m[k]) / 2 for k in tail_m}

    for label, metrics in [('tail', tail_m), ('head', head_m), ('avg', avg_m)]:
        logger.info(f"    --- {label} ---")
        for k, v in metrics.items():
            logger.info(f"      {k:8s}: {v:.4f}")

    return {'tail': tail_m, 'head': head_m, 'avg': avg_m}


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 70)
    logger.info("FASHION — TRAINING TRANSE")
    logger.info("=" * 70)

    device = diagnose_gpu()
    train_factory, valid_factory, test_factory = load_triples_factories()
    type_index = build_type_index(train_factory)

    model = train_model(train_factory, valid_factory, device)

    # Save model
    model_path = os.path.join(RESULTS_DIR, 'transe.pt')
    torch.save(model.state_dict(), model_path)
    logger.info(f"  Model saved to: {model_path}")

    # Evaluation
    results = {}
    for mode in ['naive', 'type_constrained']:
        results[mode] = evaluate(
            model, test_factory, train_factory,
            valid_factory, type_index, mode
        )

    # Save results
    results_path = os.path.join(RESULTS_DIR, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Results saved to: {results_path}")

    # Final table
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS (averaged head+tail)")
    logger.info("=" * 70)
    header = f"{'Mode':20s} {'MRR':>8s} {'H@1':>8s} {'H@3':>8s} {'H@10':>8s} {'MR':>8s}"
    logger.info(header)
    logger.info("-" * len(header))
    for mode in ['naive', 'type_constrained']:
        avg = results[mode]['avg']
        logger.info(
            f"{mode:20s} {avg['MRR']:>8.4f} {avg['Hits@1']:>8.4f} "
            f"{avg['Hits@3']:>8.4f} {avg['Hits@10']:>8.4f} {avg['MR']:>8.1f}"
        )

    logger.info("\nDONE.")


if __name__ == "__main__":
    main()
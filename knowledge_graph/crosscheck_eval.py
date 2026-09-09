"""
Cross-check our custom evaluation against PyKEEN's built-in RankBasedEvaluator.

Purpose: verify that our naive-mode evaluation in train_kge.py produces
the same numbers as PyKEEN's standard evaluator. If they match, our
implementation is correct (and type_constrained, which differs only by a
mask, can be trusted by extension).

This script loads the TransE model already trained and saved by train_kge.py
and runs both evaluations on the same test triples.
"""

import os
import logging
import torch
import pandas as pd

from pykeen.triples import TriplesFactory
from pykeen.models import TransE
from pykeen.evaluation import RankBasedEvaluator

# Re-use functions from our main script
from train_kge import (
    KG_PATH, TASKA_VALID, TASKA_TEST,
    load_triples_factories, build_type_index, evaluate,
    HITS_AT_K,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

MODEL_WEIGHTS_PATH = os.path.join("results", "step2_kge", "transe.pt")
EMBEDDING_DIM = 200
RELATION_LABEL = 'compatible_with'

# --------------------------------------------------------------------------- #
# PyKEEN-side evaluation
# --------------------------------------------------------------------------- #

def pykeen_naive_evaluation(model, train_factory, valid_factory, test_factory):
    """
    Run PyKEEN's RankBasedEvaluator in filtered mode, restricted to triples
    of the 'compatible_with' relation, ranking against ALL entities.
    This is the reference 'naive mode' implementation.
    """
    logger.info("=" * 70)
    logger.info("PYKEEN EVALUATION (reference)")
    logger.info("=" * 70)

    relation_id = train_factory.relation_to_id[RELATION_LABEL]

    # Filter test triples to only compatible_with
    test_triples = test_factory.mapped_triples
    rel_mask = test_triples[:, 1] == relation_id
    cw_test_triples = test_triples[rel_mask]
    logger.info(f"  Test triples (compatible_with only): {len(cw_test_triples):,}")

    # Build a filtered TriplesFactory containing only compatible_with triples
    cw_test_factory = TriplesFactory(
        mapped_triples=cw_test_triples,
        entity_to_id=train_factory.entity_to_id,
        relation_to_id=train_factory.relation_to_id,
    )

    # Same for valid (used as 'known' for filtering)
    valid_triples = valid_factory.mapped_triples
    cw_valid_triples = valid_triples[valid_triples[:, 1] == relation_id]

    # Run evaluator
    evaluator = RankBasedEvaluator(filtered=True)

    logger.info("  Running PyKEEN RankBasedEvaluator (filtered)...")
    results = evaluator.evaluate(
        model=model,
        mapped_triples=cw_test_factory.mapped_triples,
        additional_filter_triples=[
            train_factory.mapped_triples,
            cw_valid_triples,
        ],
        batch_size=256,
    )

    # Extract metrics in the format we care about (averaged head+tail, realistic)
    metrics_dict = results.to_dict()

    # PyKEEN's metric structure: metrics_dict['both']['realistic'][metric_name]
    # 'both' = average head+tail, 'realistic' = standard rank computation
    out = {}
    realistic = metrics_dict['both']['realistic']
    out['MRR'] = realistic['inverse_harmonic_mean_rank']  # this IS MRR
    out['MR'] = realistic['arithmetic_mean_rank']
    for k in HITS_AT_K:
        out[f'Hits@{k}'] = realistic[f'hits_at_{k}']

    return out

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    logger.info("=" * 70)
    logger.info("CROSS-CHECK: our evaluation vs PyKEEN's reference")
    logger.info("=" * 70)

    # ---- Load data ----
    train_factory, valid_factory, test_factory = load_triples_factories()
    type_index = build_type_index(train_factory)

    # ---- Load model ----
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Model weights not found at {MODEL_WEIGHTS_PATH}. "
            f"Run train_kge.py first."
        )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"  Loading TransE model from {MODEL_WEIGHTS_PATH}")
    logger.info(f"  Device: {device}")

    model = TransE(
        triples_factory=train_factory,
        embedding_dim=EMBEDDING_DIM,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.eval()

    # ---- Run PyKEEN evaluation ----
    pykeen_results = pykeen_naive_evaluation(
        model, train_factory, valid_factory, test_factory
    )

    # ---- Run our custom evaluation in naive mode ----
    logger.info("=" * 70)
    logger.info("OUR EVALUATION (custom)")
    logger.info("=" * 70)
    our_results_full = evaluate(
        model, test_factory, train_factory, valid_factory,
        type_index, mode='naive',
    )
    our_results = our_results_full['avg']  # averaged head+tail

    # ---- Compare ----
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON: PyKEEN vs Our custom evaluation (naive mode)")
    logger.info("=" * 70)
    header = f"  {'Metric':10s} {'PyKEEN':>12s} {'Ours':>12s} {'Diff':>12s} {'Rel.diff':>10s}"
    logger.info(header)
    logger.info("  " + "-" * (len(header) - 2))

    max_rel_diff = 0.0
    for metric in ['MRR', 'MR', 'Hits@1', 'Hits@3', 'Hits@10']:
        pk = pykeen_results[metric]
        ours = our_results[metric]
        diff = ours - pk
        rel = abs(diff) / abs(pk) if pk != 0 else 0.0
        max_rel_diff = max(max_rel_diff, rel)
        logger.info(
            f"  {metric:10s} {pk:>12.6f} {ours:>12.6f} "
            f"{diff:>+12.6f} {rel*100:>9.2f}%"
        )

    logger.info("\n" + "=" * 70)
    if max_rel_diff < 0.01:  # within 1%
        logger.info(f"  PASS: max relative difference = {max_rel_diff*100:.3f}%")
        logger.info("  Our custom evaluation matches PyKEEN's reference.")
    elif max_rel_diff < 0.05:  # within 5%
        logger.warning(f"  WARN: max relative difference = {max_rel_diff*100:.3f}%")
        logger.warning("  Small discrepancy -- may be due to tie-breaking strategy.")
    else:
        logger.error(f"  FAIL: max relative difference = {max_rel_diff*100:.3f}%")
        logger.error("  Significant discrepancy -- our evaluation may have a bug.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
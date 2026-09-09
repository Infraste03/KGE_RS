import os
import sys
import json
import logging
import torch
import pandas as pd

from pykeen.triples import TriplesFactory
from pykeen.models import TransE


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
BASE_DIR = os.path.abspath(os.path.join(PARENT_DIR, ".."))

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

KG_TRAIN = os.path.join(BASE_DIR, "data", "processed", "kg_train.tsv")
KG_VALID = os.path.join(BASE_DIR, "data", "processed", "taskA_valid.tsv")
KG_TEST = os.path.join(BASE_DIR, "data", "processed", "taskA_test.tsv")

HPO_DIR = os.path.join(BASE_DIR, "results", "hpo_transe")

BEST_MODEL_PATH = os.path.join(HPO_DIR, "best_model.pt")

# Possible names, because the B2B and Fashion scripts do not always use the same convention
BEST_METRICS_CANDIDATES = [
    os.path.join(HPO_DIR, "best_metrics.json"),
    os.path.join(HPO_DIR, "best_metricsTransE.json"),
    os.path.join(BASE_DIR, "best_metrics.json"),
]

BEST_PARAMS_CANDIDATES = [
    os.path.join(HPO_DIR, "best_params.json"),
    os.path.join(HPO_DIR, "best_paramsTransE.json"),
    os.path.join(BASE_DIR, "best_params.json"),
    os.path.join(BASE_DIR, "best_paramsTransE.json"),
]

RELATION_LABEL = "compatible_with"
HITS_AT_K = [1, 3, 10]

# If best_metrics.json is not available, use this value as a minimum check.
# Set it to None if no comparison with an expected log value is desired.
EXPECTED_TYPE_CONSTRAINED_MRR_FROM_LOG = 0.1057

# For quick debugging, this can be set, for example, to 100.
# For the actual verification it must remain None.
MAX_EVAL_TRIPLES = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def first_existing_path(paths):
    """Return the first existing path from a list, or None."""
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def read_triples(path: str) -> pd.DataFrame:
    """Read a TSV triple file with columns head, relation, tail."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, sep="\t")
    required = {"head", "relation", "tail"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"The file {path} does not contain the expected columns {required}. "
            f"Columns found: {list(df.columns)}"
        )
    return df


def load_factories():
    """
    Load TriplesFactory objects as in Step 2:
      - train creates the entity/relation space
      - valid/test reuse the train entity_to_id and relation_to_id mappings
    """
    train_df = read_triples(KG_TRAIN)
    valid_df = read_triples(KG_VALID)
    test_df = read_triples(KG_TEST)

    train_factory = TriplesFactory.from_labeled_triples(
        train_df[["head", "relation", "tail"]].values
    )
    valid_factory = TriplesFactory.from_labeled_triples(
        valid_df[["head", "relation", "tail"]].values,
        entity_to_id=train_factory.entity_to_id,
        relation_to_id=train_factory.relation_to_id,
    )
    test_factory = TriplesFactory.from_labeled_triples(
        test_df[["head", "relation", "tail"]].values,
        entity_to_id=train_factory.entity_to_id,
        relation_to_id=train_factory.relation_to_id,
    )

    if RELATION_LABEL not in train_factory.relation_to_id:
        raise ValueError(
            f"Relation '{RELATION_LABEL}' not found in the training KG. "
            f"Available relations: {train_factory.relation_to_id}"
        )

    return train_factory, valid_factory, test_factory


def build_type_index_fashion(train_factory):
    """
    Build the type_index for the Fashion KG.

    Expected types:
        - item
        - category
        - brand

    For compatible_with:
        - head must be item
        - tail must be item
    """
    type_to_ids = {
        "item": [],
        "category": [],
        "brand": [],
    }

    unknown = []

    for label, eid in train_factory.entity_to_id.items():
        prefix = label.split("_")[0]
        if prefix in type_to_ids:
            type_to_ids[prefix].append(eid)
        else:
            unknown.append(label)

    if unknown:
        raise ValueError(
            f"Found {len(unknown)} entities with an unknown prefix. "
            f"Sample: {unknown[:10]}"
        )

    type_index = {
        k: torch.LongTensor(sorted(v))
        for k, v in type_to_ids.items()
    }

    for k, v in type_index.items():
        if v.numel() == 0:
            raise ValueError(f"type_index['{k}'] is empty.")

    return type_index


def verify_compatible_with_is_item_item(factory, type_index, split_name):
    """
    Verify that all compatible_with triples are item -> item.
    This is the most important semantic check for Fashion.
    """
    relation_id = factory.relation_to_id[RELATION_LABEL]
    item_ids = set(type_index["item"].tolist())

    triples = factory.mapped_triples
    triples = triples[triples[:, 1] == relation_id]

    bad = []
    for h, r, t in triples.tolist():
        if h not in item_ids or t not in item_ids:
            bad.append((h, r, t))
            if len(bad) >= 5:
                break

    if bad:
        raise AssertionError(
            f"In {split_name}, some compatible_with triples are not item->item. "
            f"Examples: {bad}"
        )

    logger.info(
        f"  PASS: {split_name} compatible_with is item->item "
        f"({len(triples):,} triples)"
    )


def load_state_dict(path, device):
    """
    Load best_model.pt robustly.

    Supports:
      - direct PyKEEN state_dict
      - dict with a 'state_dict' key
      - model object with a state_dict() method
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"TransE checkpoint not found: {path}")

    obj = torch.load(path, map_location=device, weights_only=False)

    expected_keys = {
        "entity_representations.0._embeddings.weight",
        "relation_representations.0._embeddings.weight",
    }

    if isinstance(obj, dict) and expected_keys.issubset(set(obj.keys())):
        return obj

    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
        if expected_keys.issubset(set(sd.keys())):
            return sd

    if hasattr(obj, "state_dict"):
        sd = obj.state_dict()
        if expected_keys.issubset(set(sd.keys())):
            return sd

    raise ValueError(
        "Unrecognized checkpoint format. "
        f"Object type: {type(obj)}. "
        f"Available keys: {list(obj.keys()) if isinstance(obj, dict) else 'N/A'}"
    )


def infer_embedding_dim_from_state_dict(state_dict):
    """Infer embedding_dim from the entity embedding matrix."""
    entity_w = state_dict["entity_representations.0._embeddings.weight"]
    relation_w = state_dict["relation_representations.0._embeddings.weight"]

    if entity_w.ndim != 2:
        raise ValueError(f"entity_w must be 2D, found shape: {tuple(entity_w.shape)}")
    if relation_w.ndim != 2:
        raise ValueError(f"relation_w must be 2D, found shape: {tuple(relation_w.shape)}")
    if entity_w.shape[1] != relation_w.shape[1]:
        raise ValueError(
            f"Entity dim != relation dim: {entity_w.shape[1]} vs {relation_w.shape[1]}"
        )

    return entity_w.shape[1]


def maybe_read_embedding_dim_from_params():
    """
    Read embedding_dim from the best_params file, if available.
    If it does not exist, return None.
    """
    params_path = first_existing_path(BEST_PARAMS_CANDIDATES)
    if params_path is None:
        logger.warning("  No best_params.json found: using the dimension from the checkpoint.")
        return None, None

    with open(params_path, "r") as f:
        params = json.load(f)

    embedding_dim = params.get("embedding_dim", None)

    if embedding_dim is None:
        logger.warning(
            f"  Found {params_path}, but it does not contain 'embedding_dim'. "
            f"Using the dimension from the checkpoint."
        )
        return None, params_path

    return int(embedding_dim), params_path


def compute_metrics_from_ranks(ranks):
    """Compute MRR, MR, and Hits@K."""
    if len(ranks) == 0:
        raise ValueError("Empty ranks list: no triples to evaluate.")

    ranks = torch.tensor(ranks, dtype=torch.float)

    metrics = {
        "MRR": (1.0 / ranks).mean().item(),
        "MR": ranks.mean().item(),
    }

    for k in HITS_AT_K:
        metrics[f"Hits@{k}"] = (ranks <= k).float().mean().item()

    return metrics


def evaluate_custom(
    model,
    test_factory,
    train_factory,
    valid_factory,
    type_index,
    mode,
    relation_label=RELATION_LABEL,
):
    """
    Filtered evaluation on compatible_with.

    mode='naive':
        rank against all entities.

    mode='type_constrained':
        in Fashion, rank only against items:
            - tail candidates = item
            - head candidates = item
    """
    if mode not in {"naive", "type_constrained"}:
        raise ValueError(f"Invalid mode: {mode}")

    device = next(model.parameters()).device
    relation_id = train_factory.relation_to_id[relation_label]
    num_entities = train_factory.num_entities

    if mode == "type_constrained":
        item_ids = type_index["item"].to(device)

        tail_mask = torch.full((num_entities,), float("-inf"), device=device)
        tail_mask[item_ids] = 0.0

        head_mask = torch.full((num_entities,), float("-inf"), device=device)
        head_mask[item_ids] = 0.0
    else:
        tail_mask = torch.zeros(num_entities, device=device)
        head_mask = torch.zeros(num_entities, device=device)

    # Filtered evaluation:
    # remove other known positives for the same head/tail.
    head_to_known_tails = {}
    tail_to_known_heads = {}

    for fac in [train_factory, valid_factory, test_factory]:
        for h, r, t in fac.mapped_triples.tolist():
            if r != relation_id:
                continue
            head_to_known_tails.setdefault(h, set()).add(t)
            tail_to_known_heads.setdefault(t, set()).add(h)

    test_triples = test_factory.mapped_triples
    test_triples = test_triples[test_triples[:, 1] == relation_id]

    if MAX_EVAL_TRIPLES is not None:
        logger.warning(
            f"  DEBUG MODE: evaluating only the first {MAX_EVAL_TRIPLES} triples. "
            f"The metrics are NOT comparable with best_metrics.json."
        )
        test_triples = test_triples[:MAX_EVAL_TRIPLES]

    logger.info(
        f"  Evaluation mode={mode}: {len(test_triples):,} compatible_with triples"
    )

    model.eval()
    tail_ranks = []
    head_ranks = []

    with torch.no_grad():
        for i, (h, r, t) in enumerate(test_triples.tolist(), start=1):

            # ------------------------------------------------------------- #
            # Tail prediction: rank t given (h, r)
            # ------------------------------------------------------------- #
            scores = model.score_t(
                torch.tensor([[h, r]], dtype=torch.long, device=device)
            ).squeeze(0)

            scores = scores + tail_mask

            for known_tail in head_to_known_tails.get(h, set()):
                if known_tail != t:
                    scores[known_tail] = float("-inf")

            tail_rank = (scores > scores[t]).sum().item() + 1
            tail_ranks.append(tail_rank)

            # ------------------------------------------------------------- #
            # Head prediction: rank h given (r, t)
            # ------------------------------------------------------------- #
            scores = model.score_h(
                torch.tensor([[r, t]], dtype=torch.long, device=device)
            ).squeeze(0)

            scores = scores + head_mask

            for known_head in tail_to_known_heads.get(t, set()):
                if known_head != h:
                    scores[known_head] = float("-inf")

            head_rank = (scores > scores[h]).sum().item() + 1
            head_ranks.append(head_rank)

            if i % 1000 == 0:
                logger.info(f"    Processed {i:,}/{len(test_triples):,} triples...")

    tail_m = compute_metrics_from_ranks(tail_ranks)
    head_m = compute_metrics_from_ranks(head_ranks)
    avg_m = {k: (tail_m[k] + head_m[k]) / 2 for k in tail_m}

    return {
        "tail": tail_m,
        "head": head_m,
        "avg": avg_m,
    }


def extract_reported_metric(reported, mode, metric):
    """
    Extract a metric from best_metrics.json.

    Expected format, as in B2B:
        reported[mode]['avg'][metric]

    Also supports slightly simpler formats:
        reported[mode][metric]
        reported['avg'][metric]
        reported[metric]
    """
    candidates = []

    if isinstance(reported, dict):
        if mode in reported and isinstance(reported[mode], dict):
            if "avg" in reported[mode] and isinstance(reported[mode]["avg"], dict):
                candidates.append(reported[mode]["avg"])
            candidates.append(reported[mode])

        if "avg" in reported and isinstance(reported["avg"], dict):
            candidates.append(reported["avg"])

        candidates.append(reported)

    for d in candidates:
        if metric in d:
            return float(d[metric])

    return None


def print_results_table(results):
    """Print the tail/head/avg results table."""
    print()
    print(f"  {'Mode':18s} {'Metric':10s} {'Tail':>10s} {'Head':>10s} {'Avg':>10s}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for mode, mode_res in results.items():
        for metric in ["MRR", "MR", "Hits@1", "Hits@3", "Hits@10"]:
            print(
                f"  {mode:18s} "
                f"{metric:10s} "
                f"{mode_res['tail'][metric]:>10.4f} "
                f"{mode_res['head'][metric]:>10.4f} "
                f"{mode_res['avg'][metric]:>10.4f}"
            )
        print()


def compare_with_best_metrics(results):
    """
    Compare reproduced results with best_metrics.json, if available.
    If unavailable, use EXPECTED_TYPE_CONSTRAINED_MRR_FROM_LOG.
    """
    metrics_path = first_existing_path(BEST_METRICS_CANDIDATES)

    if metrics_path is None:
        logger.warning("No best_metrics.json found.")

        if EXPECTED_TYPE_CONSTRAINED_MRR_FROM_LOG is not None:
            reproduced = results["type_constrained"]["avg"]["MRR"]
            expected = EXPECTED_TYPE_CONSTRAINED_MRR_FROM_LOG
            diff = abs(reproduced - expected)

            print()
            print("=" * 70)
            print("COMPARISON WITH EXPECTED MRR FROM LOG")
            print("=" * 70)
            print(f"  Expected MRR from log     : {expected:.4f}")
            print(f"  Reproduced MRR            : {reproduced:.4f}")
            print(f"  Absolute difference       : {diff:.4f}")

            if diff < 0.005:
                print("  VERDICT: PASS — checkpoint is consistent with the log.")
            elif diff < 0.02:
                print("  VERDICT: WARNING — small difference, check seed/log.")
            else:
                print("  VERDICT: FAIL — checkpoint probably does not match the log.")
            print("=" * 70)

        return

    with open(metrics_path, "r") as f:
        reported = json.load(f)

    print()
    print("=" * 70)
    print("COMPARISON WITH best_metrics.json")
    print("=" * 70)
    print(f"  File: {metrics_path}")
    print()
    print(f"  {'Metric':10s} {'Mode':18s} {'Reported':>12s} {'Reproduced':>12s} {'Diff':>12s}")
    print(f"  {'-'*10} {'-'*18} {'-'*12} {'-'*12} {'-'*12}")

    max_diff = 0.0
    found_any = False

    for mode in ["naive", "type_constrained"]:
        for metric in ["MRR", "Hits@1", "Hits@3", "Hits@10"]:
            reported_value = extract_reported_metric(reported, mode, metric)

            if reported_value is None:
                continue

            reproduced_value = results[mode]["avg"][metric]
            diff = reproduced_value - reported_value
            max_diff = max(max_diff, abs(diff))
            found_any = True

            print(
                f"  {metric:10s} "
                f"{mode:18s} "
                f"{reported_value:>12.4f} "
                f"{reproduced_value:>12.4f} "
                f"{diff:>+12.4f}"
            )

    if not found_any:
        print("  WARNING: unable to read comparable metrics from the JSON file.")
        print("  Check the structure of best_metrics.json.")
        return

    print()
    print(f"  Max absolute difference: {max_diff:.4f}")

    if max_diff < 0.005:
        print("  VERDICT: PASS — best_model.pt reproduces best_metrics.json.")
    elif max_diff < 0.02:
        print("  VERDICT: WARNING — small numerical difference.")
    else:
        print("  VERDICT: FAIL — best_model.pt may not be the correct checkpoint.")

    print("=" * 70)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    logger.info("=" * 70)
    logger.info("VERIFY STEP 2 METRICS — FASHION")
    logger.info("=" * 70)

    # ------------------------------------------------------------------- #
    # 1. Load TriplesFactories
    # ------------------------------------------------------------------- #
    logger.info("\n[1/5] Loading TriplesFactory objects...")
    train_factory, valid_factory, test_factory = load_factories()

    logger.info(f"  Entities   : {train_factory.num_entities:,}")
    logger.info(f"  Relations  : {train_factory.num_relations}")
    logger.info(f"  Train      : {train_factory.num_triples:,}")
    logger.info(f"  Valid      : {valid_factory.num_triples:,}")
    logger.info(f"  Test       : {test_factory.num_triples:,}")
    logger.info(f"  relation_to_id: {train_factory.relation_to_id}")

    # ------------------------------------------------------------------- #
    # 2. Build type index and semantic checks
    # ------------------------------------------------------------------- #
    logger.info("\n[2/5] Building Fashion type_index...")
    type_index = build_type_index_fashion(train_factory)

    for entity_type, ids in type_index.items():
        logger.info(f"  {entity_type:10s}: {ids.numel():,}")

    logger.info("\n  Checking compatible_with item->item semantics...")
    verify_compatible_with_is_item_item(train_factory, type_index, "train")
    verify_compatible_with_is_item_item(valid_factory, type_index, "valid")
    verify_compatible_with_is_item_item(test_factory, type_index, "test")

    # ------------------------------------------------------------------- #
    # 3. Load checkpoint and infer/check embedding_dim
    # ------------------------------------------------------------------- #
    logger.info("\n[3/5] Loading TransE checkpoint...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"  Device: {device}")
    logger.info(f"  Checkpoint: {BEST_MODEL_PATH}")

    state_dict = load_state_dict(BEST_MODEL_PATH, device=device)

    ckpt_embedding_dim = infer_embedding_dim_from_state_dict(state_dict)
    params_embedding_dim, params_path = maybe_read_embedding_dim_from_params()

    if params_path is not None:
        logger.info(f"  best_params: {params_path}")

    if params_embedding_dim is not None:
        logger.info(f"  embedding_dim from best_params : {params_embedding_dim}")
        logger.info(f"  embedding_dim from checkpoint  : {ckpt_embedding_dim}")

        if params_embedding_dim != ckpt_embedding_dim:
            raise ValueError(
                f"embedding_dim mismatch: best_params={params_embedding_dim}, "
                f"checkpoint={ckpt_embedding_dim}"
            )

    embedding_dim = ckpt_embedding_dim

    entity_w = state_dict["entity_representations.0._embeddings.weight"]
    relation_w = state_dict["relation_representations.0._embeddings.weight"]

    if entity_w.shape[0] != train_factory.num_entities:
        raise ValueError(
            f"Entity count mismatch: checkpoint={entity_w.shape[0]}, "
            f"TriplesFactory={train_factory.num_entities}"
        )

    if relation_w.shape[0] != train_factory.num_relations:
        raise ValueError(
            f"Relation count mismatch: checkpoint={relation_w.shape[0]}, "
            f"TriplesFactory={train_factory.num_relations}"
        )

    logger.info(f"  Entity weights shape   : {tuple(entity_w.shape)}")
    logger.info(f"  Relation weights shape : {tuple(relation_w.shape)}")

    # ------------------------------------------------------------------- #
    # 4. Build TransE model and load weights
    # ------------------------------------------------------------------- #
    logger.info("\n[4/5] Rebuilding PyKEEN TransE model...")

    model = TransE(
        triples_factory=train_factory,
        embedding_dim=embedding_dim,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        ent_w = model.entity_representations[0]._embeddings.weight
        rel_w = model.relation_representations[0]._embeddings.weight

        ent_norms = ent_w.norm(dim=1)
        rel_norms = rel_w.norm(dim=1)

        logger.info(
            f"  Entity norms   : mean={ent_norms.mean().item():.4f}, "
            f"std={ent_norms.std().item():.4f}, "
            f"min={ent_norms.min().item():.4f}, "
            f"max={ent_norms.max().item():.4f}"
        )
        logger.info(
            f"  Relation norms : mean={rel_norms.mean().item():.4f}, "
            f"std={rel_norms.std().item():.4f}"
        )

        if torch.isnan(ent_w).any() or torch.isnan(rel_w).any():
            raise ValueError("NaN values found in model weights.")
        if torch.isinf(ent_w).any() or torch.isinf(rel_w).any():
            raise ValueError("Inf values found in model weights.")

    # ------------------------------------------------------------------- #
    # 5. Evaluate in both modes
    # ------------------------------------------------------------------- #
    logger.info("\n[5/5] Evaluating Step 2...")

    results = {}

    logger.info("\n--- mode=naive ---")
    results["naive"] = evaluate_custom(
        model=model,
        test_factory=test_factory,
        train_factory=train_factory,
        valid_factory=valid_factory,
        type_index=type_index,
        mode="naive",
    )

    logger.info("\n--- mode=type_constrained ---")
    results["type_constrained"] = evaluate_custom(
        model=model,
        test_factory=test_factory,
        train_factory=train_factory,
        valid_factory=valid_factory,
        type_index=type_index,
        mode="type_constrained",
    )

    logger.info("\nReproduced results:")
    print_results_table(results)

    compare_with_best_metrics(results)

    logger.info("Verification completed.")


if __name__ == "__main__":
    main()
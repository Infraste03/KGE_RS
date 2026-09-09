"""
eval_utils_fashion.py
======================

Evaluation functions used for monitoring during Fashion alternate learning.

evaluate_task_a_fashion : MRR and Hits@K on compatible_with (link prediction)
evaluate_task_b_fashion : Recall@20 and NDCG@20 on RecBole valid/test splits

CRITICAL DIFFERENCES compared with B2B (eval_utils.py):

  evaluate_task_a:
    - type_index uses 'item' candidates for compatible_with
    - compatible_with is item->item: type-constrained and naive evaluation coincide
      for this relation
    - Known-Answer Filtering is included for correct filtered evaluation

  evaluate_task_b:
    - scores are indexed by POSITION in the candidate vector [0..165468],
      NOT by RecBole ID as in B2B
    - candidate_ids = torch.arange(ITEM_ID_START, NUM_KG_ENTITIES) — items only
    - Masking excludes range(seq_len-1), i.e. training items, but NOT the
      validation item
      (verified empirically in test_task_b_real_data_fashion.py:
       Recall@20 difference < 0.001 compared with the official RecBole Trainer)
    - RecBole -> candidate-position translation for masking:
      pos = r2u[recbole_id] - ITEM_ID_START
    - Fashion padding = 182983 (not 0): RecBole padding is skipped with sid == 0

File location:
    fashion_generalization/alternate_learning/eval_utils_fashion.py
"""

import torch
import logging
from math import log2
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(THIS_DIR, "models")

if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from joint_model_fashion import JointAlternateModelFashion

logger = logging.getLogger(__name__)

# Fashion constants
ITEM_ID_START   = 17_514
NUM_KG_ENTITIES = 182_983   # real KG entities (0..182982)
UNIFIED_PADDING = 182_983   # unified padding (= NUM_KG_ENTITIES)
NUM_ITEMS_REAL  = 165_469   # = NUM_KG_ENTITIES - ITEM_ID_START

# Diagnostic flag: printed only for the first evaluation batch
_EVAL_TASK_B_DEBUG_PRINTED = False


# =========================================================================== #
# Task A — Link Prediction (MRR, Hits@K)
# =========================================================================== #

def evaluate_task_a_fashion(
    model,
    kg_data_dict: dict,
    type_index: dict,
    relation_label: str = 'compatible_with',
    batch_size_eval: int = 128,
    filtered: bool = True,
) -> dict:
    """
    Compute MRR and Hits@K for Task A (TransE) on compatible_with.

    Replicates the logic used in test_task_a_real_data_fashion.py:
    - Type-constrained: items only as head and tail candidates
    - Filtered: exclude known triples (train+valid+test) from ranking
    - Average avg(tail_MRR, head_MRR), following PyKEEN convention

    Parameters
    ----------
    model : JointAlternateModelFashion
    kg_data_dict : dict
        Output of load_kg_fashion. Contains train/valid/test_triples,
        entity_to_id, relation_to_id, num_entities.
    type_index : dict
        {'item': LongTensor, 'category': LongTensor, 'brand': LongTensor}
        Produced by the type-index construction used in Fashion checks.
    relation_label : str
        Relation to evaluate. Default: 'compatible_with'.
    batch_size_eval : int
        Number of triples per batch. Reduce to 64 if OOM occurs.
    filtered : bool
        If True (default), apply Known-Answer Filtering as in PyKEEN.

    Returns
    -------
    dict with 'MRR', 'Hits@1', 'Hits@3', 'Hits@10'
    """
    logger.info(
        f"  [Eval Task A] relation='{relation_label}', filtered={filtered}"
    )
    model.eval()
    device = next(model.parameters()).device

    relation_id = kg_data_dict['relation_to_id'].get(relation_label)

    if relation_id is None:
        logger.warning(
            f"  Relation '{relation_label}' not found in the KG."
        )
        return {
            "MRR": 0.0,
            "Hits@1": 0.0,
            "Hits@3": 0.0,
            "Hits@10": 0.0,
        }

    # Filter test triples for the selected relation
    test_triples = torch.as_tensor(
        kg_data_dict['test_triples'],
        dtype=torch.long
    )

    mask = test_triples[:, 1] == relation_id
    test_triples = test_triples[mask]

    if len(test_triples) == 0:
        logger.warning(
            f"  No test triples found for '{relation_label}'."
        )
        return {
            "MRR": 0.0,
            "Hits@1": 0.0,
            "Hits@3": 0.0,
            "Hits@10": 0.0,
        }

    logger.info(
        f"  compatible_with test triples: {len(test_triples):,}"
    )

    # Type mask: only items are valid candidates
    # (compatible_with is item->item)
    num_entities = kg_data_dict['num_entities']
    item_ids = type_index['item'].to(device)

    type_mask = torch.full(
        (num_entities,),
        float('-inf'),
        device=device
    )
    type_mask[item_ids] = 0.0

    # Known-Answer Filter:
    # build h -> known_tails and t -> known_heads dictionaries
    if filtered:
        head_to_known_tails = {}
        tail_to_known_heads = {}

        for split in [
            'train_triples',
            'valid_triples',
            'test_triples'
        ]:
            triples = torch.as_tensor(
                kg_data_dict[split],
                dtype=torch.long
            )

            triples = triples[
                triples[:, 1] == relation_id
            ]

            for h, r, t in triples.tolist():
                head_to_known_tails.setdefault(
                    h,
                    set()
                ).add(t)

                tail_to_known_heads.setdefault(
                    t,
                    set()
                ).add(h)

    h_all = test_triples[:, 0].to(device)
    r_all = test_triples[:, 1].to(device)
    t_all = test_triples[:, 2].to(device)

    num_triples = len(h_all)

    tail_ranks = []
    head_ranks = []

    with torch.no_grad():

        for start in range(
            0,
            num_triples,
            batch_size_eval
        ):
            end = min(
                start + batch_size_eval,
                num_triples
            )

            h_b = h_all[start:end]
            r_b = r_all[start:end]
            t_b = t_all[start:end]

            # --- Tail prediction ---
            scores_tail = model.score_all_tails_kge(
                h_b,
                r_b
            )  # (B, num_entities+1)

            # Exclude the padding row
            scores_tail = scores_tail[:, :num_entities]

            # Apply type constraint
            scores_tail = (
                scores_tail
                + type_mask.unsqueeze(0)
            )

            if filtered:
                for i, (h, t) in enumerate(
                    zip(
                        h_b.tolist(),
                        t_b.tolist()
                    )
                ):
                    for kt in head_to_known_tails.get(
                        h,
                        set()
                    ):
                        if kt != t:
                            scores_tail[i, kt] = float('-inf')

            true_scores_t = scores_tail[
                torch.arange(len(t_b)),
                t_b
            ]

            ranks_t = (
                scores_tail
                > true_scores_t.unsqueeze(1)
            ).sum(dim=1) + 1

            tail_ranks.extend(
                ranks_t.tolist()
            )

            # --- Head prediction ---
            scores_head = model.score_all_heads_kge(
                r_b,
                t_b
            )  # (B, num_entities+1)

            scores_head = scores_head[:, :num_entities]

            scores_head = (
                scores_head
                + type_mask.unsqueeze(0)
            )

            if filtered:
                for i, (h, t) in enumerate(
                    zip(
                        h_b.tolist(),
                        t_b.tolist()
                    )
                ):
                    for kh in tail_to_known_heads.get(
                        t,
                        set()
                    ):
                        if kh != h:
                            scores_head[i, kh] = float('-inf')

            true_scores_h = scores_head[
                torch.arange(len(h_b)),
                h_b
            ]

            ranks_h = (
                scores_head
                > true_scores_h.unsqueeze(1)
            ).sum(dim=1) + 1

            head_ranks.extend(
                ranks_h.tolist()
            )

    def _metrics(ranks):
        r = torch.tensor(
            ranks,
            dtype=torch.float
        )

        return {
            'MRR': (
                1.0 / r
            ).mean().item(),

            'Hits@1': (
                r <= 1
            ).float().mean().item(),

            'Hits@3': (
                r <= 3
            ).float().mean().item(),

            'Hits@10': (
                r <= 10
            ).float().mean().item(),
        }

    tail_m = _metrics(tail_ranks)
    head_m = _metrics(head_ranks)

    avg_m = {
        k: (
            tail_m[k] + head_m[k]
        ) / 2
        for k in tail_m
    }

    logger.info(
        f"  Task A: MRR={avg_m['MRR']:.4f}, "
        f"H@1={avg_m['Hits@1']:.4f}, "
        f"H@3={avg_m['Hits@3']:.4f}, "
        f"H@10={avg_m['Hits@10']:.4f}"
    )

    return avg_m


# =========================================================================== #
# Task B — Sequential Recommendation (Recall@20, NDCG@20)
# =========================================================================== #

def evaluate_task_b_fashion(
    model,
    recbole_eval_dataloader,
    recbole_to_unified: torch.Tensor,
    topk: int = 20,
) -> dict:
    """
    Compute Recall@K and NDCG@K for Task B (Fashion SASRec).

    Replicates the protocol verified in test_task_b_real_data_fashion.py:
    - Masking excludes training items (range(seq_len-1)), but NOT the
      validation item
    - Candidates include only unified item IDs (17514..182982),
      excluding brands, categories, and padding
    - Scores are indexed by position in the candidate vector,
      NOT by RecBole ID

    IMPORTANT — difference compared with B2B:
    In B2B:
        scores shape (B, num_items_recbole), indexed by RecBole ID

    In Fashion:
        scores shape (B, 165469), indexed by position in [17514..182982]

    Masking therefore requires:
        RecBole ID -> unified ID -> candidate position

    Parameters
    ----------
    model : JointAlternateModelFashion

    recbole_eval_dataloader : RecBole DataLoader (valid or test)
        Yields tuples (interaction, _, _, _).

        interaction['item_id_list'] : RecBole IDs, padding=0
        interaction['item_length']  : real sequence lengths
        interaction['item_id']      : target RecBole ID

    recbole_to_unified : LongTensor (num_items_recbole,)
        RecBole -> unified ID translation tensor.
        Position 0 -> 182983 (padding).

    topk : int
        Default: 20.

    Returns
    -------
    dict containing 'Recall@20' and 'NDCG@20'
    """
    global _EVAL_TASK_B_DEBUG_PRINTED

    logger.info(
        f"  [Eval Task B] Recall@{topk} and NDCG@{topk}..."
    )

    model.eval()
    device = next(model.parameters()).device

    r2u = recbole_to_unified.to(device)

    # Candidate unified IDs: items only (17514..182982)
    candidate_ids = torch.arange(
        ITEM_ID_START,
        NUM_KG_ENTITIES,
        dtype=torch.long,
        device=device
    )  # shape (165469,)

    hits_count = 0
    ndcg_sum   = 0.0
    total      = 0

    with torch.no_grad():

        for batch_data in recbole_eval_dataloader:

            interaction, _, _, _ = batch_data

            item_seq_recbole = interaction[
                'item_id_list'
            ].to(device)

            item_seq_len = interaction[
                'item_length'
            ].to(device)

            target_recbole = interaction[
                'item_id'
            ].to(device)

            # Translate RecBole sequence -> unified IDs
            item_seq_unified = r2u[
                item_seq_recbole
            ]

            # Forward pass
            user_repr = model.forward_sasrec(
                item_seq_unified,
                item_seq_len
            )

            # scores shape: (B, 165469)
            # position i corresponds to unified ID (17514 + i)
            scores = model.score_items_sasrec(
                user_repr,
                candidate_ids
            )

            # Masking diagnostic
            # (first batch, first user only)
            if not _EVAL_TASK_B_DEBUG_PRINTED:
                _log_masking_diagnostic(
                    scores,
                    item_seq_recbole,
                    item_seq_len,
                    target_recbole,
                    r2u,
                    device
                )
                _EVAL_TASK_B_DEBUG_PRINTED = True

            for b in range(
                user_repr.shape[0]
            ):
                seq_len = item_seq_len[
                    b
                ].item()

                target_rid = target_recbole[
                    b
                ].item()

                target_uid = r2u[
                    target_rid
                ].item()

                target_pos = (
                    target_uid
                    - ITEM_ID_START
                )

                if (
                    target_pos < 0
                    or target_pos >= scores.shape[1]
                ):
                    # Target is not an item.
                    # This should never occur.
                    total += 1
                    continue

                scores_b = scores[
                    b
                ].clone()

                # Mask training items:
                # positions 0..seq_len-2
                #
                # Do NOT mask position seq_len-1
                # because it corresponds to the validation item.
                #
                # This reproduces the verified RecBole protocol.
                for pos in range(
                    seq_len - 1
                ):
                    sid = item_seq_recbole[
                        b,
                        pos
                    ].item()

                    if sid == 0:
                        # RecBole padding
                        continue

                    seen_uid = r2u[
                        sid
                    ].item()

                    seen_pos = (
                        seen_uid
                        - ITEM_ID_START
                    )

                    if (
                        0 <= seen_pos < scores_b.shape[0]
                        and seen_pos != target_pos
                    ):
                        scores_b[
                            seen_pos
                        ] = float('-inf')

                target_score = scores_b[
                    target_pos
                ].item()

                rank = (
                    scores_b
                    > target_score
                ).sum().item() + 1

                if rank <= topk:
                    hits_count += 1
                    ndcg_sum += (
                        1.0
                        / log2(rank + 1)
                    )

                total += 1

    recall = (
        hits_count / total
        if total > 0
        else 0.0
    )

    ndcg = (
        ndcg_sum / total
        if total > 0
        else 0.0
    )

    # Mathematical sanity check:
    # in Leave-One-Out evaluation, NDCG@K <= Recall@K
    assert ndcg <= recall + 1e-6, \
        f"Inconsistency: NDCG@{topk}={ndcg:.4f} > Recall@{topk}={recall:.4f}"

    logger.info(
        f"  Task B: Recall@{topk}={recall:.4f}, "
        f"NDCG@{topk}={ndcg:.4f}"
    )

    return {
        f"Recall@{topk}": recall,
        f"NDCG@{topk}": ndcg
    }


def _log_masking_diagnostic(
    scores,
    item_seq_recbole,
    item_seq_len,
    target_recbole,
    r2u,
    device
):
    """
    Print a detailed masking diagnostic for the first user.

    Executed only once during evaluation.
    """
    b = 0

    seq_len = item_seq_len[
        b
    ].item()

    target_rid = target_recbole[
        b
    ].item()

    target_uid = r2u[
        target_rid
    ].item()

    target_pos = (
        target_uid
        - ITEM_ID_START
    )

    logger.info("=" * 70)
    logger.info(
        "TASK B EVALUATION MASKING DIAGNOSTIC "
        "(executed once)"
    )

    logger.info(
        f"  seq_len={seq_len}, "
        f"target RecBole ID={target_rid}, "
        f"target unified ID={target_uid}, "
        f"target_pos={target_pos}"
    )

    logger.info(
        f"  RecBole sequence (first 10): "
        f"{item_seq_recbole[b, :10].tolist()}"
    )

    # Verify masking for up to five training items
    # (positions 0..seq_len-2)
    logger.info(
        f"  Training items to mask "
        f"(positions 0..{seq_len-2}):"
    )

    masked_ok = 0

    for pos in range(
        min(seq_len - 1, 5)
    ):
        sid = item_seq_recbole[
            b,
            pos
        ].item()

        if sid == 0:
            continue

        seen_uid = r2u[
            sid
        ].item()

        seen_pos = (
            seen_uid
            - ITEM_ID_START
        )

        if 0 <= seen_pos < scores.shape[1]:
            score_pre = scores[
                b,
                seen_pos
            ].item()

            score_post = float('-inf')

            logger.info(
                f"    pos={pos} "
                f"RecBoleID={sid} "
                f"unifiedID={seen_uid} "
                f"candidatePos={seen_pos} "
                f"score_pre={score_pre:.4f} "
                f"-> score_post=-inf OK"
            )

            masked_ok += 1

    # Validation item (position seq_len-1):
    # it must NOT be masked
    if seq_len >= 1:
        val_sid = item_seq_recbole[
            b,
            seq_len - 1
        ].item()

        val_uid = (
            r2u[val_sid].item()
            if val_sid != 0
            else UNIFIED_PADDING
        )

        val_pos = (
            val_uid
            - ITEM_ID_START
        )

        if 0 <= val_pos < scores.shape[1]:
            score_val = scores[
                b,
                val_pos
            ].item()

            logger.info(
                f"  Validation item "
                f"(position {seq_len-1}): "
                f"RecBoleID={val_sid} "
                f"unifiedID={val_uid} "
                f"score={score_val:.4f} "
                f"(NOT masked — correct)"
            )

    # Target item score
    if 0 <= target_pos < scores.shape[1]:
        score_target = scores[
            b,
            target_pos
        ].item()

        logger.info(
            f"  Target item: "
            f"RecBoleID={target_rid} "
            f"unifiedID={target_uid} "
            f"pos={target_pos} "
            f"score={score_target:.4f}"
        )

    logger.info(
        f"  Training items checked: {masked_ok}"
    )

    logger.info("=" * 70)


def reset_eval_debug():
    """
    Reset the diagnostic flag.

    Useful when validation and test evaluation are executed separately.
    """
    global _EVAL_TASK_B_DEBUG_PRINTED
    _EVAL_TASK_B_DEBUG_PRINTED = False


# =========================================================================== #
# SMOKE TEST
# =========================================================================== #

if __name__ == "__main__":

    import os
    import sys

    THIS_DIR = os.path.dirname(
        os.path.abspath(__file__)
    )

    if THIS_DIR not in sys.path:
        sys.path.insert(
            0,
            THIS_DIR
        )

    print("=" * 70)
    print("EVAL UTILS FASHION — SMOKE TEST")
    print("=" * 70)

    # Real Fashion dimensions
    # (evaluation examples are reduced for speed)
    NUM_ENTITIES    = 182_984
    NUM_RELATIONS   = 3
    KGE_DIM         = 64
    UNIFIED_PADDING = 182_983
    ITEM_ID_START_  = 17_514
    NUM_KG_ENT      = 182_983
    BATCH           = 4
    SEQ_LEN         = 10

    model = JointAlternateModelFashion(
        num_entities=NUM_ENTITIES,
        num_relations=NUM_RELATIONS,
        kge_dim=KGE_DIM,
        unified_padding_idx=UNIFIED_PADDING,
    )

    model.eval()

    # ------------------------------------------------------------------ #
    # TEST 1: evaluate_task_a_fashion with fake data
    # ------------------------------------------------------------------ #
    print(
        "\n--- Test 1: evaluate_task_a_fashion ---"
    )

    # Minimal fake KG:
    # compatible_with relation between items only
    #
    # item unified IDs: 17514..17520 (7 items)
    ITEM_START = ITEM_ID_START_

    fake_train = torch.tensor([
        [
            ITEM_START,
            0,
            ITEM_START + 1
        ],
        [
            ITEM_START + 1,
            0,
            ITEM_START + 2
        ],
        [
            ITEM_START + 2,
            0,
            ITEM_START + 3
        ],
    ], dtype=torch.long)

    fake_valid = torch.tensor([
        [
            ITEM_START + 3,
            0,
            ITEM_START + 4
        ],
    ], dtype=torch.long)

    fake_test = torch.tensor([
        [
            ITEM_START + 4,
            0,
            ITEM_START + 5
        ],
        [
            ITEM_START + 5,
            0,
            ITEM_START + 6
        ],
    ], dtype=torch.long)

    fake_kg = {
        'train_triples': fake_train.numpy(),
        'valid_triples': fake_valid.numpy(),
        'test_triples': fake_test.numpy(),
        'relation_to_id': {
            'compatible_with': 0
        },
        'num_entities': NUM_KG_ENT,
    }

    # type_index: all fake item unified IDs
    fake_type_index = {
        'item': torch.arange(
            ITEM_START,
            ITEM_START + 7,
            dtype=torch.long
        ),
    }

    metrics_a = evaluate_task_a_fashion(
        model=model,
        kg_data_dict=fake_kg,
        type_index=fake_type_index,
        relation_label='compatible_with',
        batch_size_eval=4,
        filtered=True,
    )

    assert 'MRR' in metrics_a, \
        "MRR missing from output"

    assert 'Hits@1' in metrics_a, \
        "Hits@1 missing from output"

    assert 'Hits@3' in metrics_a, \
        "Hits@3 missing from output"

    assert 'Hits@10' in metrics_a, \
        "Hits@10 missing from output"

    assert 0.0 <= metrics_a['MRR'] <= 1.0, \
        f"MRR out of range: {metrics_a['MRR']}"

    assert 0.0 <= metrics_a['Hits@10'] <= 1.0, \
        f"H@10 out of range: {metrics_a['Hits@10']}"

    print(
        f"  MRR={metrics_a['MRR']:.4f}, "
        f"H@1={metrics_a['Hits@1']:.4f}, "
        f"H@3={metrics_a['Hits@3']:.4f}, "
        f"H@10={metrics_a['Hits@10']:.4f}"
    )

    print(
        f"  PASS: evaluate_task_a_fashion"
    )

    # ------------------------------------------------------------------ #
    # TEST 2: evaluate_task_b_fashion with fake DataLoader
    # ------------------------------------------------------------------ #
    print(
        "\n--- Test 2: evaluate_task_b_fashion ---"
    )

    # r2u tensor:
    # RecBole 0 -> 182983
    # RecBole 1..8 -> unified item IDs
    r2u = torch.full(
        (9,),
        UNIFIED_PADDING,
        dtype=torch.long
    )

    for i in range(1, 8):
        r2u[i] = (
            ITEM_START + i - 1
        )

    # Mock RecBole evaluation DataLoader
    # yielding tuples like the real DataLoader
    class MockEvalLoader:

        def __iter__(self):

            # Batch with 4 users
            #
            # item_seq: RecBole IDs, padding=0
            # seq_len: includes training items + validation item
            # item_id: target RecBole ID (test item)
            interaction = {
                'user_id': torch.tensor([
                    1,
                    2,
                    3,
                    4
                ]),

                'item_id_list': torch.tensor([
                    [
                        1, 2, 3, 0, 0,
                        0, 0, 0, 0, 0
                    ],
                    [
                        2, 3, 4, 0, 0,
                        0, 0, 0, 0, 0
                    ],
                    [
                        1, 3, 0, 0, 0,
                        0, 0, 0, 0, 0
                    ],
                    [
                        4, 5, 6, 7, 0,
                        0, 0, 0, 0, 0
                    ],
                ]),

                'item_length': torch.tensor([
                    2,
                    3,
                    2,
                    4
                ]),

                'item_id': torch.tensor([
                    4,
                    5,
                    5,
                    1
                ]),
            }

            yield (
                interaction,
                None,
                None,
                None
            )

    # Ensure that the diagnostic is printed
    reset_eval_debug()

    metrics_b = evaluate_task_b_fashion(
        model=model,
        recbole_eval_dataloader=MockEvalLoader(),
        recbole_to_unified=r2u,
        topk=20,
    )

    assert 'Recall@20' in metrics_b, \
        "Recall@20 missing"

    assert 'NDCG@20' in metrics_b, \
        "NDCG@20 missing"

    assert 0.0 <= metrics_b['Recall@20'] <= 1.0, \
        f"Recall@20 out of range: {metrics_b['Recall@20']}"

    assert 0.0 <= metrics_b['NDCG@20'] <= 1.0, \
        f"NDCG@20 out of range: {metrics_b['NDCG@20']}"

    assert (
        metrics_b['NDCG@20']
        <= metrics_b['Recall@20'] + 1e-6
    ), "NDCG > Recall — mathematical inconsistency!"

    print(
        f"  Recall@20={metrics_b['Recall@20']:.4f}, "
        f"NDCG@20={metrics_b['NDCG@20']:.4f}"
    )

    print(
        f"  PASS: evaluate_task_b_fashion"
    )

    # ------------------------------------------------------------------ #
    # TEST 3: correct masking — validation item is NOT masked
    # ------------------------------------------------------------------ #
    print(
        "\n--- Test 3: masking verification — "
        "validation item is not masked ---"
    )

    # Controlled example:
    #
    # user sequence = [item_A, item_B], seq_len=2
    #
    # training item   = item_A (position 0, must be masked)
    # validation item = item_B (position 1, must NOT be masked)
    # target          = item_C (test item)
    ITEM_A = ITEM_START
    ITEM_B = ITEM_START + 1
    ITEM_C = ITEM_START + 2

    r2u_test3 = torch.full(
        (5,),
        UNIFIED_PADDING,
        dtype=torch.long
    )

    r2u_test3[1] = ITEM_A
    r2u_test3[2] = ITEM_B
    r2u_test3[3] = ITEM_C

    class MockLoaderTest3:

        def __iter__(self):

            interaction = {
                'user_id': torch.tensor([
                    1
                ]),

                'item_id_list': torch.tensor([[
                    1, 2, 0, 0, 0,
                    0, 0, 0, 0, 0
                ]]),

                'item_length': torch.tensor([
                    2
                ]),

                'item_id': torch.tensor([
                    3
                ]),
            }

            yield (
                interaction,
                None,
                None,
                None
            )

    # Run with model in evaluation mode and random weights
    model.eval()

    candidate_ids = torch.arange(
        ITEM_START,
        ITEM_START + 7,
        dtype=torch.long
    )

    with torch.no_grad():

        seq_u = r2u_test3[
            torch.tensor([[
                1, 2, 0, 0, 0,
                0, 0, 0, 0, 0
            ]])
        ]

        user_repr = model.forward_sasrec(
            seq_u,
            torch.tensor([2])
        )

        scores_all = model.score_items_sasrec(
            user_repr,
            candidate_ids
        )

    # Position 0 = ITEM_A
    # training item, must be masked
    #
    # Position 1 = ITEM_B
    # validation item, must remain available
    #
    # Position 2 = ITEM_C
    # target item, must remain available
    scores_b = scores_all[
        0
    ].clone()

    seq_len = 2

    for pos in range(
        seq_len - 1
    ):
        sid = 1

        seen_uid = r2u_test3[
            sid
        ].item()

        seen_pos = (
            seen_uid
            - ITEM_START
        )

        scores_b[
            seen_pos
        ] = float('-inf')

    # ITEM_A must be masked
    assert scores_b[0].item() == float('-inf'), \
        f"ITEM_A (training) was not masked! score={scores_b[0].item()}"

    print(
        f"  ITEM_A (training, pos=0): "
        f"score=-inf (correctly masked)"
    )

    # ITEM_B must NOT be masked
    assert scores_b[1].item() != float('-inf'), \
        "ITEM_B (validation) was incorrectly masked!"

    print(
        f"  ITEM_B (validation, pos=1): "
        f"score={scores_b[1].item():.4f} "
        f"(NOT masked)"
    )

    # ITEM_C target must NOT be masked
    assert scores_b[2].item() != float('-inf'), \
        "ITEM_C (target) was incorrectly masked!"

    print(
        f"  ITEM_C (target, pos=2): "
        f"score={scores_b[2].item():.4f} "
        f"(NOT masked)"
    )

    print(
        f"  PASS: masking is correct"
    )

    # ------------------------------------------------------------------ #
    # TEST 4: reset_eval_debug
    # ------------------------------------------------------------------ #
    print(
        "\n--- Test 4: reset_eval_debug ---"
    )

    reset_eval_debug()

    assert not _EVAL_TASK_B_DEBUG_PRINTED, \
        "Diagnostic flag was not reset!"

    print(
        f"  PASS: diagnostic flag reset correctly"
    )

    print("\n" + "=" * 70)
    print("EVAL UTILS FASHION — SMOKE TEST PASSED")
    print("=" * 70)
    print()
    print("Fashion evaluation protocol summary:")
    print(
        f"  Task A: filtered MRR on "
        f"compatible_with (item->item)"
    )
    print(
        f"  Task B: Recall@20, NDCG@20 "
        f"with masking range(seq_len-1)"
    )
    print(
        f"          candidates: unified IDs "
        f"[{ITEM_ID_START_}..{NUM_KG_ENT-1}]"
    )
    print(
        f"          RecBole padding (0) -> "
        f"{UNIFIED_PADDING} "
        f"(skipped during masking)"
    )
"""
eval_utils_fashion_v3.py
========================

Evaluation functions used to monitor Fashion alternate learning v3.

evaluate_task_a_fashion : MRR and Hits@K for compatible_with link prediction
evaluate_task_b_fashion : Recall@20 and NDCG@20 on RecBole validation/test data

CRITICAL DIFFERENCES compared with B2B (eval_utils.py):

  evaluate_task_a:
    - type_index uses only 'item' candidates for compatible_with
    - compatible_with is item -> item
    - Known-Answer Filtering is included for filtered link-prediction evaluation

  evaluate_task_b:
    - scores are indexed by POSITION in the candidate vector, not by RecBole ID
    - candidate_ids contains only real Fashion item unified IDs
    - masking excludes range(seq_len - 1), i.e. training items, but DOES NOT
      exclude the validation item
    - RecBole IDs are translated to candidate positions through:
          pos = r2u[recbole_id] - ITEM_ID_START
    - RecBole padding ID 0 is ignored during masking

Fashion v3 item range:
    ITEM_ID_START   = 9,317
    ITEM_ID_END     = 81,251
    NUM_ITEMS_REAL  = 71,935
    UNIFIED_PADDING = 81,259

Location:
    fashion_generalization/alternate_learning/variants/v3/eval_utils_fashion_v3.py
"""

import torch
import logging
from math import log2
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# This file is stored under:
#   alternate_learning/variants/v3/
#
# Main model modules are stored under:
#   alternate_learning/models/
ALTERNATE_DIR = os.path.abspath(
    os.path.join(THIS_DIR, "..", "..")
)

MODELS_DIR = os.path.join(
    ALTERNATE_DIR,
    "models"
)

if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from joint_model_fashion import JointAlternateModelFashion

logger = logging.getLogger(__name__)


# Fashion v3 constants
ITEM_ID_START   = 9_317
NUM_KG_ENTITIES = 81_252    # Exclusive upper bound for item IDs (81,251 + 1)
                             # IMPORTANT: in v3, items are not the final
                             # alphabetically ordered entity block.
                             # poptier/pricetier entities occur after the last
                             # item and before padding. Therefore,
                             # NUM_KG_ENTITIES != UNIFIED_PADDING.
UNIFIED_PADDING = 81_259    # Actual unified padding ID (= total number of entities)
NUM_ITEMS_REAL  = 71_935    # 81,252 - 9,317

# Diagnostic flag: printed only for the first Task B evaluation batch
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
    Compute MRR and Hits@K for Fashion Task A (TransE) on compatible_with.

    The evaluation follows the protocol used in
    test_task_a_real_data_fashion.py:

    - Type-constrained evaluation: only items are candidate heads/tails.
    - Filtered evaluation: known triples are excluded from ranking.
    - Final metrics average tail-prediction and head-prediction metrics,
      following the PyKEEN convention.

    Parameters
    ----------
    model : JointAlternateModelFashion
    kg_data_dict : dict
        Output from load_kg_fashion. Contains train/valid/test_triples,
        entity_to_id, relation_to_id, and num_entities.
    type_index : dict
        Dictionary containing entity IDs grouped by type.
    relation_label : str
        Relation to evaluate. Default: 'compatible_with'.
    batch_size_eval : int
        Number of triples per evaluation batch. Reduce to 64 in case of OOM.
    filtered : bool
        If True, apply Known-Answer Filtering as in PyKEEN.

    Returns
    -------
    dict containing 'MRR', 'Hits@1', 'Hits@3', and 'Hits@10'.
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

    # Type mask: only items are valid candidates for compatible_with
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

            # Tail prediction
            scores_tail = model.score_all_tails_kge(
                h_b,
                r_b
            )

            # Exclude the padding row
            scores_tail = scores_tail[
                :,
                :num_entities
            ]

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
                            scores_tail[
                                i,
                                kt
                            ] = float('-inf')

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

            # Head prediction
            scores_head = model.score_all_heads_kge(
                r_b,
                t_b
            )

            scores_head = scores_head[
                :,
                :num_entities
            ]

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
                            scores_head[
                                i,
                                kh
                            ] = float('-inf')

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
            tail_m[k]
            + head_m[k]
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
    Compute Recall@K and NDCG@K for Fashion Task B (SASRec).

    The evaluation follows the protocol validated in
    test_task_b_real_data_fashion.py:

    - Masking excludes training items through range(seq_len - 1).
    - The validation item is NOT masked.
    - Candidates contain only real Fashion items.
    - Scores are indexed by candidate-vector position, not by RecBole ID.

    Fashion v3 candidates correspond to unified IDs:

        9,317 .. 81,251

    Therefore the score matrix contains 71,935 candidate positions.

    Candidate masking requires the following translation:

        RecBole ID -> unified ID -> candidate position

        candidate_position = unified_id - ITEM_ID_START

    Parameters
    ----------
    model : JointAlternateModelFashion

    recbole_eval_dataloader : RecBole DataLoader
        Validation or test DataLoader.

        Each batch contains:
            interaction['item_id_list'] : RecBole IDs, padding=0
            interaction['item_length']  : real sequence lengths
            interaction['item_id']      : target RecBole ID

    recbole_to_unified : LongTensor (num_items_recbole,)
        Translation tensor from RecBole IDs to unified IDs.
        Position 0 corresponds to RecBole padding and maps to
        UNIFIED_PADDING.

    topk : int
        Default: 20.

    Returns
    -------
    dict containing Recall@K and NDCG@K.
    """
    global _EVAL_TASK_B_DEBUG_PRINTED

    logger.info(
        f"  [Eval Task B] Recall@{topk} and NDCG@{topk}..."
    )

    model.eval()
    device = next(model.parameters()).device

    r2u = recbole_to_unified.to(device)

    # Candidate unified IDs: only real Fashion items
    candidate_ids = torch.arange(
        ITEM_ID_START,
        NUM_KG_ENTITIES,
        dtype=torch.long,
        device=device
    )

    hits_count = 0
    ndcg_sum = 0.0
    total = 0

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

            # Translate RecBole sequence IDs -> unified IDs
            item_seq_unified = r2u[
                item_seq_recbole
            ]

            # Forward pass
            user_repr = model.forward_sasrec(
                item_seq_unified,
                item_seq_len
            )

            # Score positions correspond to candidate unified IDs
            scores = model.score_items_sasrec(
                user_repr,
                candidate_ids
            )

            # Masking diagnostic for the first user of the first batch only
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
                    # The target is not a valid item.
                    # This should never happen with the expected dataset.
                    total += 1
                    continue

                scores_b = scores[
                    b
                ].clone()

                # Mask training items at positions 0..seq_len-2.
                #
                # Do NOT mask position seq_len-1 because it is the
                # validation item. This reproduces the verified
                # RecBole evaluation protocol.
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
    # under Leave-One-Out evaluation, NDCG@K <= Recall@K
    assert ndcg <= recall + 1e-6, \
        (
            f"Inconsistent metrics: "
            f"NDCG@{topk}={ndcg:.4f} > "
            f"Recall@{topk}={recall:.4f}"
        )

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
    Log a detailed masking diagnostic for the first user.

    This diagnostic is executed only once during evaluation.
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

    # Check masking for up to five training items
    logger.info(
        f"  Training items to mask "
        f"(positions 0..{seq_len - 2}):"
    )

    masked_ok = 0

    for pos in range(
        min(
            seq_len - 1,
            5
        )
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

        if (
            0 <= seen_pos < scores.shape[1]
        ):
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

    # The validation item at position seq_len-1 must NOT be masked
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

        if (
            0 <= val_pos < scores.shape[1]
        ):
            score_val = scores[
                b,
                val_pos
            ].item()

            logger.info(
                f"  Validation item "
                f"(position {seq_len - 1}): "
                f"RecBoleID={val_sid} "
                f"unifiedID={val_uid} "
                f"score={score_val:.4f} "
                f"(NOT masked — correct)"
            )

    # Target item score
    if (
        0 <= target_pos < scores.shape[1]
    ):
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
        f"  Verified training items: "
        f"{masked_ok}"
    )

    logger.info("=" * 70)


def reset_eval_debug():
    """
    Reset the evaluation diagnostic flag.

    Useful when validation and test evaluation should each print
    their own first-batch diagnostic.
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

    # Real Fashion dimensions from the previous setup,
    # reduced where appropriate for faster testing
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
    # TEST 1: evaluate_task_a_fashion with synthetic data
    # ------------------------------------------------------------------ #

    print(
        "\n--- Test 1: evaluate_task_a_fashion ---"
    )

    # Minimal synthetic KG containing only compatible_with between items
    # Item unified IDs: 17514..17520
    ITEM_START = ITEM_ID_START_

    fake_train = torch.tensor([
        [ITEM_START,     0, ITEM_START + 1],
        [ITEM_START + 1, 0, ITEM_START + 2],
        [ITEM_START + 2, 0, ITEM_START + 3],
    ], dtype=torch.long)

    fake_valid = torch.tensor([
        [ITEM_START + 3, 0, ITEM_START + 4],
    ], dtype=torch.long)

    fake_test = torch.tensor([
        [ITEM_START + 4, 0, ITEM_START + 5],
        [ITEM_START + 5, 0, ITEM_START + 6],
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

    # type_index contains all synthetic item unified IDs
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
        "  PASS: evaluate_task_a_fashion"
    )

    # ------------------------------------------------------------------ #
    # TEST 2: evaluate_task_b_fashion with a synthetic DataLoader
    # ------------------------------------------------------------------ #

    print(
        "\n--- Test 2: evaluate_task_b_fashion ---"
    )

    # r2u tensor:
    # RecBole 0 -> 182983
    # RecBole IDs 1..8 -> synthetic unified item IDs
    r2u = torch.full(
        (9,),
        UNIFIED_PADDING,
        dtype=torch.long
    )

    for i in range(
        1,
        8
    ):
        r2u[i] = ITEM_START + i - 1

    # Mock RecBole evaluation DataLoader.
    # It emits tuples as the real evaluation DataLoader does.
    class MockEvalLoader:

        def __iter__(self):

            # Batch with four users.
            #
            # item_seq: RecBole IDs, padding=0
            # seq_len: includes training items + validation item
            # item_id : target RecBole ID (test item)
            interaction = {
                'user_id': torch.tensor([
                    1, 2, 3, 4
                ]),

                'item_id_list': torch.tensor([
                    [1, 2, 3, 0, 0, 0, 0, 0, 0, 0],
                    [2, 3, 4, 0, 0, 0, 0, 0, 0, 0],
                    [1, 3, 0, 0, 0, 0, 0, 0, 0, 0],
                    [4, 5, 6, 7, 0, 0, 0, 0, 0, 0],
                ]),

                'item_length': torch.tensor([
                    2, 3, 2, 4
                ]),

                'item_id': torch.tensor([
                    4, 5, 5, 1
                ]),
            }

            yield (
                interaction,
                None,
                None,
                None
            )

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
        "  PASS: evaluate_task_b_fashion"
    )

    # ------------------------------------------------------------------ #
    # TEST 3: masking check — validation item must NOT be masked
    # ------------------------------------------------------------------ #

    print(
        "\n--- Test 3: masking check — validation item is not masked ---"
    )

    # Controlled example:
    # sequence = [item_A, item_B], seq_len=2
    #
    # training item   = item_A, position 0 -> must be masked
    # validation item = item_B, position 1 -> must NOT be masked
    # target          = item_C
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
                'user_id': torch.tensor([1]),

                'item_id_list': torch.tensor([
                    [1, 2, 0, 0, 0, 0, 0, 0, 0, 0]
                ]),

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

    # Run with the randomly initialized model in evaluation mode
    model.eval()

    candidate_ids = torch.arange(
        ITEM_START,
        ITEM_START + 7,
        dtype=torch.long
    )

    with torch.no_grad():

        seq_u = r2u_test3[
            torch.tensor([
                [1, 2, 0, 0, 0, 0, 0, 0, 0, 0]
            ])
        ]

        user_repr = model.forward_sasrec(
            seq_u,
            torch.tensor([2])
        )

        scores_all = model.score_items_sasrec(
            user_repr,
            candidate_ids
        )

    # Position 0 = ITEM_A (training)
    # Position 1 = ITEM_B (validation)
    # Position 2 = ITEM_C (target)
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
        (
            f"ITEM_A (training) was not masked! "
            f"score={scores_b[0].item()}"
        )

    print(
        "  ITEM_A (training, pos=0): "
        "score=-inf OK (correctly masked)"
    )

    # ITEM_B must not be masked
    assert scores_b[1].item() != float('-inf'), \
        "ITEM_B (validation) was incorrectly masked!"

    print(
        f"  ITEM_B (validation, pos=1): "
        f"score={scores_b[1].item():.4f} "
        f"OK (NOT masked)"
    )

    # ITEM_C target must not be masked
    assert scores_b[2].item() != float('-inf'), \
        "ITEM_C (target) was incorrectly masked!"

    print(
        f"  ITEM_C (target, pos=2): "
        f"score={scores_b[2].item():.4f} "
        f"OK (NOT masked)"
    )

    print(
        "  PASS: masking is correct"
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
        "  PASS: diagnostic flag reset correctly"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "EVAL UTILS FASHION — SMOKE TEST PASSED"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Fashion evaluation protocol summary:"
    )

    print(
        "  Task A: filtered MRR on compatible_with "
        "(item -> item)"
    )

    print(
        "  Task B: Recall@20 and NDCG@20 "
        "with range(seq_len - 1) masking"
    )

    print(
        f"          candidates: unified IDs "
        f"[{ITEM_ID_START_}..{NUM_KG_ENT - 1}]"
    )

    print(
        f"          RecBole padding (0) -> "
        f"{UNIFIED_PADDING} "
        f"(skipped during masking)"
    )
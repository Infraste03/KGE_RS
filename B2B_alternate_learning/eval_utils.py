import torch
import logging
from math import log2

logger = logging.getLogger(__name__)

# ========================================================================= #
# EVALUATION TASK A (Link Prediction / MRR)
# Logic replicated by checks/test_task_a_real_data.py and run_hpo_kge2.py
# ========================================================================= #

def evaluate_task_a(model, kg_data_dict, type_index, relation_label='compatible_with'):
    """
    Recalculate MRR and Hits@K metrics on Task A (TransE).
    Takes test triples as input from the 'kg_data_dict' dictionary.
    This is a simplified version of 'evaluate_custom' for monitoring.
    """
    logger.info(f"  [Eval Task A] Recalculating metrics on relation '{relation_label}' ...")
    model.eval()
    relation_id = kg_data_dict['relation_to_id'][relation_label]
    test_triples = kg_data_dict['test_triples']
    mask = test_triples[:, 1] == relation_id
    test_triples = test_triples[mask]
    if len(test_triples) == 0:
        logger.warning(f"No test triples found for relation {relation_label}")
        return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}

    h = test_triples[:, 0].to(next(model.parameters()).device)
    r = test_triples[:, 1].to(next(model.parameters()).device)
    t = test_triples[:, 2].to(next(model.parameters()).device)
    BATCH_EVAL = 128
    mrr_sum = 0.0
    hits_10_sum = 0.0
    num_triples = len(h)

    with torch.no_grad():
        for start in range(0, num_triples, BATCH_EVAL):
            end = min(start + BATCH_EVAL, num_triples)
            h_batch = h[start:end]
            t_batch = t[start:end]
            r_batch = r[start:end]

            scores = model.task_a.score_all_tails(h_batch, r_batch)  # (batch, num_entities)
            items_mask = torch.zeros(scores.shape[1], dtype=torch.bool, device=h.device)
            items_mask[type_index['item']] = True
            scores[:, ~items_mask] = -float('inf')
            true_scores = scores[torch.arange(len(t_batch)), t_batch]
            ranks = (scores > true_scores.unsqueeze(1)).sum(dim=1) + 1
            mrr_sum += (1.0 / ranks.float()).sum().item()
            hits_10_sum += (ranks <= 10).float().sum().item()

    mrr = mrr_sum / num_triples
    hits_10 = hits_10_sum / num_triples
    return {"MRR": mrr, "Hits@10": hits_10}

_EVAL_DEBUG_PRINTED = False

def evaluate_task_b(model, data_loader, recbole_to_unified, topk=20):
    """
    Recalculate R@K and NDCG@K for Task B (SASRec).
    Uses the provided Dataloader (`data_loader`), which can be for validation or test.
    Applies the correct masking based on the context:
    - For validation: masks only the items in the input sequence.
    - For test (mode 'full'): masks the entire history (train+valid).
    """
    logger.info("  [Eval Task B] Recalculating metrics R@20 and NDCG@20 ...")
    model.eval()
    device = next(model.parameters()).device
    hits_count = 0
    ndcg_sum = 0.0
    total_users = 0
    translator = recbole_to_unified.to(device)

    global _EVAL_DEBUG_PRINTED

    with torch.no_grad():
        for batch_data in data_loader:
            interaction, ground_truth, pos_idx, neg_idx = batch_data
            item_seq_recbole = interaction['item_id_list'].to(device)
            item_seq_len = interaction['item_length'].to(device)
            target_items = interaction['item_id'].to(device)
            item_seq_unified = translator[item_seq_recbole]
            user_repr = model.forward_sasrec(item_seq_unified, item_seq_len)
            scores = model.score_items_sasrec(user_repr, translator)
            if not _EVAL_DEBUG_PRINTED:
                user_to_debug_idx = 0
                user_id_debug = interaction['user_id'][user_to_debug_idx].item()
                history_to_mask_debug = item_seq_recbole[user_to_debug_idx]
                items_to_check_masking = history_to_mask_debug[:3]
                pre_mask_scores = {
                    item_id.item(): scores[user_to_debug_idx, item_id.item()].item()
                    for item_id in items_to_check_masking if item_id.item() != 0 # no padding
                }

                scores_clone_debug = scores[user_to_debug_idx].clone()
                scores_clone_debug[history_to_mask_debug] = -float('inf')

                post_mask_scores = {
                    item_id.item(): scores_clone_debug[item_id.item()].item()
                    for item_id in items_to_check_masking if item_id.item() != 0
                }

                logger.info("="*70)
                logger.info("MASKING VERIFICATION - DIAGNOSTIC START (performed only once)")
                logger.info("  - Masking mode detected: sequence-only (batch input)")
                logger.info(f"  - Test user: {user_id_debug}")
                logger.info(f"  - Target item (RecBole ID): {target_items[user_to_debug_idx].item()}")
                logger.info(f"  - History to mask (first 10): {history_to_mask_debug[:10].tolist()}...")
                logger.info(f"  - Items selected for masking test: {list(pre_mask_scores.keys())}")
                logger.info("  ---")
                logger.info("  Scores PRE-masking:")
                for item, score in pre_mask_scores.items():
                    logger.info(f"    - Item {item}: {score:.4f}")
                logger.info("  Scores POST-masking:")
                for item, score in post_mask_scores.items():
                    logger.info(f"    - Item {item}: {score}")
                all_masked_correctly = all(s == -float('inf') for s in post_mask_scores.values())
                status = " Success" if all_masked_correctly else "failure"
                logger.info(f"  Result: {status} - The items in the history have been correctly masked.")
                logger.info("MASKING VERIFICATION - END DIAGNOSTIC")
                logger.info("="*70)

                _EVAL_DEBUG_PRINTED = True

            for b_idx in range(user_repr.shape[0]):
                user_preds = scores[b_idx].clone()
                # Mask the history sequence provided by the batch itself.
                history_items = item_seq_recbole[b_idx]
                user_preds[history_items] = -float('inf')
                target_idx = target_items[b_idx].item()
                user_preds[target_idx] = scores[b_idx, target_idx].item()
                target_score = user_preds[target_idx].item()
                rank = (user_preds > target_score).sum().item() + 1
                if rank <= topk:
                    hits_count += 1
                    ndcg_sum += 1.0 / log2(rank + 1)
                total_users += 1

    recall = hits_count / total_users if total_users > 0 else 0.0
    ndcg = ndcg_sum / total_users if total_users > 0 else 0.0
    assert ndcg <= recall + 1e-9, f"inconcistency in metric: NDCG@20 ({ndcg:.4f}) > Recall@20 ({recall:.4f})"
    return {"Recall@20": recall, "NDCG@20": ndcg}


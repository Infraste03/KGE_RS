import torch
import logging
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_trainer, init_seed, init_logger
from recbole.model.sequential_recommender.sasrec import SASRec

# Import our evaluation function
from B2B_alternate_learning.eval_utils import evaluate_task_b
from B2B_alternate_learning.unified_id_space import build_unified_id_space


# Settings
MODEL_PATH = "SASRec/pth/SASRec-May-19-2026_08-28-06.pth"
USER_ID_TO_TEST = 111 # The same user as in the previous log

def get_recbole_topk(model, test_data, user_id, topk=20):
    """Use RecBole's internal methods to get recommendations."""
    model.eval()
    # Find user interaction in test_data
    for batch_data in test_data:
        interaction, _, _, _ = batch_data
        user_ids_in_batch = interaction['user_id'].tolist()
        if user_id in user_ids_in_batch:
            batch_idx = user_ids_in_batch.index(user_id)

            scores = model.full_sort_predict(interaction.to(model.device)) # (B, n_items)

            user_scores = scores[batch_idx]
            _, topk_items = torch.topk(user_scores, k=topk, dim=0)

            return topk_items.tolist()
    return None

def get_our_topk(joint_model, test_data, recbole_to_unified, user_id, topk=20):
    """Use our evaluate_task_b function to get recommendations."""
    joint_model.eval()
    device = next(joint_model.parameters()).device
    translator = recbole_to_unified.to(device)
    is_full_mode = hasattr(test_data.dataset, 'user_history_dict')

    for batch_data in test_data:
        interaction, _, _, _ = batch_data
        user_ids_in_batch = interaction['user_id'].tolist()
        if user_id in user_ids_in_batch:
            batch_idx = user_ids_in_batch.index(user_id)

            item_seq_recbole = interaction['item_id_list'].to(device)
            item_seq_len = interaction['item_length'].to(device)

            item_seq_unified = translator[item_seq_recbole]
            user_repr = joint_model.forward_sasrec(item_seq_unified, item_seq_len)
            scores = joint_model.score_items_sasrec(user_repr, translator)
            user_scores = scores[batch_idx].clone()

            if is_full_mode:
                history_items = test_data.dataset.user_history_dict[user_id]
            else:
                history_items = item_seq_recbole[batch_idx]

            user_scores[history_items] = -float('inf')

            _, topk_items = torch.topk(user_scores, k=topk, dim=0)
            return topk_items.tolist()
    return None


if __name__ == '__main__':
    config = Config(
    model='SASRec',
    dataset='b2b_data',
    config_file_list=['B2B_alternate_learning/configs/step4_config.yaml'],
    config_dict={
        'load_col': {
            'inter': ['user_id', 'item_id', 'timestamp']
        },
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'TIME_FIELD': 'timestamp',
        'loss_type': 'BPR',
        'train_neg_sample_args': {
            'distribution': 'uniform',
            'sample_num': 1
        }
    }
)
    init_seed(config['seed'], config['reproducibility'])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    original_sasrec = SASRec(config, train_data.dataset).to(config['device'])
    #original_sasrec.load_state_dict(torch.load(MODEL_PATH, weights_only=False))
    full_checkpoint = torch.load(MODEL_PATH, weights_only=False)
    original_sasrec.load_state_dict(full_checkpoint['state_dict'])

    print("--- 1. Recommendations from RecBole (official) ---")
    recbole_recs = get_recbole_topk(original_sasrec, test_data, USER_ID_TO_TEST)
    print(f"Top 20 for user {USER_ID_TO_TEST}:")
    print(recbole_recs)
    print("-" * 50)
    from B2B_alternate_learning.models.joint_model import JointAlternateModel
    from B2B_alternate_learning.models.shared_embedding import SharedEmbedding

space = build_unified_id_space(
    kg_train_path="data/processed/kg_train.tsv",
    kg_valid_path="data/processed/taskA_valid.tsv",
    kg_test_path="data/processed/taskA_test.tsv",
    recbole_train_inter="dataset/b2b_data/b2b_data.train.inter",
    recbole_valid_inter="dataset/b2b_data/b2b_data.valid.inter",
    recbole_test_inter="dataset/b2b_data/b2b_data.test.inter",
    recbole_dataset_name="b2b_data",
    recbole_data_path="dataset",
    verbose=False,
)

num_recbole_items = dataset.num('item')
recbole_to_unified = torch.zeros(num_recbole_items, dtype=torch.long)
for rec_id in range(num_recbole_items):
    unified_id = space.recbole_to_unified(rec_id)
    if unified_id is not None:
        recbole_to_unified[rec_id] = unified_id
    else:

        recbole_to_unified[rec_id] = space.num_total_entities


    shared_embedding = SharedEmbedding(num_embeddings=len(space.id_maps['unified_to_entity']), embedding_dim=config['embedding_size'])
    joint_model = JointAlternateModel(config, shared_embedding, task_b_dataset=dataset)
    joint_model.load_pretrained_sasrec(MODEL_PATH)
    joint_model.to(config['device'])

    print("--- 2. Recommendations from our function `evaluate_task_b` ---")
    our_recs = get_our_topk(joint_model, test_data, recbole_to_unified, USER_ID_TO_TEST)
    print(f"Top 20 for user {USER_ID_TO_TEST}:")
    print(our_recs)
    print("-" * 50)

    # 6. Confronto finale
    print("--- 3. Result of the comparison ---")
    if recbole_recs == our_recs:
        print(" SUCCESS: The recommendation lists are IDENTICAL.")
        print("Our implementation of the masking is a correct replica of RecBole.")
    else:
        print(" FAILURE: The recommendation lists are DIFFERENT.")
        print("There is a discrepancy in the evaluation logic.")

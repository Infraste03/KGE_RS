import torch
from logging import config, getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import SASRec
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger

def evaluate_from_checkpoint(model_path, best_params):
    # =================================================================================
    # 1. CONFIGURATION
    # =================================================================================

    config_dict = {
        'model': 'SASRec',
        'dataset': 'b2b_data',
        'data_path': 'dataset/',
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'TIME_FIELD': 'timestamp',
        'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
        'field_separator': "\t",
        'eval_setting': 'TO_session,full',
        'train_file': 'b2b_data.train.inter',
        'valid_file': 'b2b_data.valid.inter',
        'test_file': 'b2b_data.test.inter',
        'metrics': ["Recall", "NDCG"],
        'topk': [20],
        'valid_metric': 'NDCG@20',
        'seed': 2020,
        'reproducibility': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }


    config_dict.update(best_params)
    config = Config(config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()

    logger.info("Configurazione caricata:")
    logger.info(config)

    # =================================================================================
    # 2. DATASET CREATION AND PREPARATION
    # =================================================================================
    dataset = create_dataset(config)
    logger.info(f"Created dataset: {dataset}")
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # =================================================================================
    # 3. MODEL INITIALIZATION AND WEIGHT LOADING
    # =================================================================================
    model = SASRec(config, train_data.dataset).to(config['device'])
    logger.info(f"Initialized model:\n{model}")

    checkpoint = torch.load(model_path, map_location=config['device'], weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    logger.info(f"Model weights loaded successfully from: {model_path}")

    # =================================================================================
    # 4. EVALUATION
    # =================================================================================

    trainer = Trainer(config, model)
    logger.info('>>> Starting evaluation on the test set...')
    test_result = trainer.evaluate(test_data, load_best_model=False)

    # =================================================================================
    # 5. PRINTING THE RESULTS
    # =================================================================================
    logger.info('>>> Evaluation results from the checkpoint:')
    logger.info(test_result)
    print("\nEvaluation results from the checkpoint:")
    print(test_result)

    return test_result

if __name__ == '__main__':


    best_sasrec_params = {
        "embedding_size": 512,
        "n_layers": 2,
        "n_heads": 8,
        "learning_rate": 0.001,
        "hidden_dropout_prob": 0.5,
        "dropout_rate": 0.3,
        "weight_decay": 1e-05,
        "max_seq_length": 300,
        'loss_type': 'BPR',
        "sampling_size": 5
    }

    best_model_path = r'SASRec-Oct-01-2025_17-39-33.pth'
    evaluate_from_checkpoint(best_model_path, best_sasrec_params)
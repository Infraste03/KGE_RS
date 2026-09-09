from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import SASRec
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
import logging
import pandas as pd
import numpy as np
import random
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


if __name__ == '__main__':

    out_folder = "SASREC400"
    os.makedirs(out_folder, exist_ok=True)
    param_space = {
        'hidden_size': [400],
        'MAX_ITEM_LIST_LENGTH': [50],
        'n_layers': [2, 4],
        'n_heads': [2, 4, 8],
        'learning_rate': np.logspace(-5, -3, 5),
        'hidden_dropout_prob': [0.3, 0.4, 0.5],
        'attn_dropout_prob': [0.1, 0.2, 0.3],
        'weight_decay': np.logspace(-5, -4, 3),
        'loss_type': ['CE', 'BPR'],
        'train_neg_sample_args': [{'candidate_num': 1}, {'candidate_num': 3}, {'candidate_num': 5}]
    }

    num_iterations = 50
    results_file = os.path.join(out_folder, "hpo_results_sasrec_400.csv")
    all_results = []
    tested_params_set = set()

    if os.path.exists(results_file):
        logging.warning(f"File not found: '{results_file}' .")
        try:
            results_df = pd.read_csv(results_file)
            all_results = results_df.to_dict('records')

            param_keys = list(param_space.keys())
            for params_dict in all_results:

                try:
                    tested_params_set.add(tuple(str(params_dict[k]) for k in param_keys))
                except KeyError:
                    pass
            logging.warning(f"Found {len(tested_params_set)} combinations already tested. They will be skipped.")
        except Exception as e:
            logging.error(f"Impossible to read the results file. Error: {e}. The search will restart from zero.")
            all_results = []
            tested_params_set = set()

    logging.info(f"STARTING HPO WITH {num_iterations} combinations.")

    i = len(all_results)
    while i < num_iterations:
        params = {k: random.choice(list(v)) if hasattr(v, '__iter__') and not isinstance(random.choice(list(v)), dict) else random.choice(list(v)) for k, v in param_space.items()}

        param_tuple = tuple(str(params[k]) for k in param_space.keys())
        if param_tuple in tested_params_set:
            logging.info("Combination already tested found, generating a new one.")
            continue


        if params['hidden_size'] % params['n_heads'] != 0:
            logging.warning(f"Combination not valid skipped: hidden_size={params['hidden_size']} not divisible by {params['n_heads']}.")
            continue

        tested_params_set.add(param_tuple)

        run_name = f"run_{i+1}_" + "_".join([f"{k}_{v:.4f}" if isinstance(v, float) else f"{k}_{str(v).replace(':', '')}" for k, v in params.items()])
        logging.info("\n" + "="*80)
        logging.info(f"execution {i+1}/{num_iterations}: {run_name}")
        logging.info(f"Parameters: {params}")
        logging.info("="*80)

        config_dict = {
            'dataset': 'b2b_data',
            'data_path': 'dataset/',
            'USER_ID_FIELD': 'user_id',
            'ITEM_ID_FIELD': 'item_id',
            'TIME_FIELD': 'timestamp',
            'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
            'field_separator': "\t",
            'train_file': 'b2b_data.train.inter',
            'valid_file': 'b2b_data.valid.inter',
            'test_file': 'b2b_data.test.inter',
            'metrics': ["Recall", "NDCG"],
            'topk': [20],
            'valid_metric': 'NDCG@20',
            'epochs': 200,
            'train_batch_size': 256,
            'eval_batch_size': 512,
            'stopping_step': 10,
            'checkpoint_dir': os.path.join(out_folder, f'saved_hpo_runs/{run_name}/'),
            'reproducibility': True,
            'seed': 2020,
            'use_gpu': True,
        }

        config_dict.update(params)
        if config_dict['loss_type'] == 'CE':
            config_dict['train_neg_sample_args'] = None
        else:
            cand = config_dict['train_neg_sample_args']['candidate_num']
            config_dict['train_neg_sample_args'] = {
                'distribution': 'uniform', 'sample_num': cand, 'alpha': 1.0, 'dynamic': False, 'candidate_num': 0
            }

        try:
            config = Config(model='SASRec', config_dict=config_dict)
            init_seed(config['seed'], config['reproducibility'])
            init_logger(config)
            logger = logging.getLogger()

            dataset = create_dataset(config)
            train_data, valid_data, test_data = data_preparation(config, dataset)

            model = SASRec(config, train_data.dataset).to(config['device'])
            trainer = Trainer(config, model)
            best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
            test_result = trainer.evaluate(test_data)

            current_result = params.copy()
            current_result['train_neg_sample_args'] = str(current_result['train_neg_sample_args'])
            current_result['NDCG@20'] = test_result.get('ndcg@20', 0)
            current_result['Recall@20'] = test_result.get('recall@20', 0)
            all_results.append(current_result)

        except Exception as e:
            logging.error(f"Error during execution with parameters {params}: {e}")
            current_result = params.copy()
            current_result['train_neg_sample_args'] = str(current_result['train_neg_sample_args'])
            current_result['NDCG@20'] = 'FAILED'
            current_result['Recall@20'] = 'FAILED'
            all_results.append(current_result)

        i += 1

        results_df = pd.DataFrame(all_results)
        results_df.to_csv(results_file, index=False)
        logging.info(f"Intermediary results saved to '{results_file}'.")

    # --- 4. FINAL ANALYSIS ---
    logging.info("\n" + "="*80)
    logging.info("results of the hyperparameter search are completed")
    logging.info("="*80)

    final_results_df = pd.read_csv(results_file)
    final_results_df = final_results_df[final_results_df['NDCG@20'] != 'FAILED']
    final_results_df['NDCG@20'] = pd.to_numeric(final_results_df['NDCG@20'])
    final_results_df = final_results_df.sort_values(by='NDCG@20', ascending=False)
    if len(final_results_df) > 0:
        best_run = final_results_df.iloc[0]
        logging.info("Best combination of hyperparameters found:")
        logging.info(best_run)
        logging.info(f"File with all results saved in: {results_file}")

        logging.info("\n--- Top 5 Runs ---")
        logging.info(final_results_df.head(5))
    else:
        logging.info("No run completed with success.")
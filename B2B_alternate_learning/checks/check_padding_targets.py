from pathlib import Path

import yaml
from recbole.config import Config as RecBoleConfig
from recbole.data import create_dataset, data_preparation


# Path to the B2B alternate learning directory
ALTERNATE_DIR = Path(__file__).resolve().parents[1]

# Configuration file
CFG_PATH = ALTERNATE_DIR / "configs" / "step4_config.yaml"

print("Loading config:", CFG_PATH)

with open(CFG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

recbole_cfg = RecBoleConfig(model='SASRec', dataset=cfg['recbole_config'].get('dataset', 'b2b_data'), config_dict=cfg['recbole_config'])
try:
    print('RecBole config seed:', recbole_cfg['seed'])
except Exception:
    print('RecBole config has no seed field')

# create dataset and dataloaders
recbole_dataset = create_dataset(recbole_cfg)
train_data, valid_data, test_data = data_preparation(recbole_cfg, recbole_dataset)

print('Datasets created. Iterating to count target==0 occurrences...')

def count_zero(loader):
    zero_count = 0
    total = 0
    for batch in loader:
        interaction = batch[0]
        targets = interaction['item_id']
        # targets may be tensor on CPU
        zero_count += (targets == 0).sum().item()
        total += targets.numel()
    return zero_count, total

z_train, t_train = count_zero(train_data)
z_valid, t_valid = count_zero(valid_data)
z_test, t_test = count_zero(test_data)

print(f"Train targets: {z_train}/{t_train} are padding (0)")
print(f"Valid targets: {z_valid}/{t_valid} are padding (0)")
print(f"Test  targets: {z_test}/{t_test} are padding (0)")

# Also check if any users in user_history_dict have padding only (?) optional
if hasattr(valid_data.dataset, 'user_history_dict'):
    uh = valid_data.dataset.user_history_dict
    zero_hist = sum(1 for k,v in uh.items() if len(v)==0)
    print('Valid user_history_dict entries with empty history:', zero_hist)

print('Done.')

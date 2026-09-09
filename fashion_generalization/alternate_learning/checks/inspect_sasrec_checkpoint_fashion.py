from pathlib import Path
import torch

FASHION_ROOT = Path(__file__).resolve().parents[2]

CKPT = FASHION_ROOT / "results" / "trial_017" / "best_valid.pth"

ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)

if isinstance(ckpt, dict) and 'state_dict' in ckpt:
    sd = ckpt['state_dict']
    print("Format: dict with key 'state_dict'")

    for k in ckpt:
        if k != 'state_dict':
            print(f"  Extra key: '{k}' = {ckpt[k]}")
else:
    sd = ckpt
    print("Format: state_dict directly")

print(f"\nAll keys in the state_dict ({len(sd)} total):")
for k, v in sorted(sd.items()):
    print(f"  {k:60s} shape={tuple(v.shape)}")

print("\n--- Parameters ---")


ie = sd['item_embedding.weight']
num_items, hidden_size = ie.shape
print(f"num_items (with PAD) : {num_items}")
print(f"hidden_size         : {hidden_size}")


pe = sd['position_embedding.weight']
max_seq_length = pe.shape[0]
print(f"max_seq_length      : {max_seq_length}")


n_layers = 0
while f'trm_encoder.layer.{n_layers}.multi_head_attention.query.weight' in sd:
    n_layers += 1
print(f"n_layers            : {n_layers}")

print(f"n_heads             : NOT readable by shapes")

# inner_size from feed_forward
inner_key = 'trm_encoder.layer.0.feed_forward.dense_1.weight'
if inner_key in sd:
    inner_size = sd[inner_key].shape[0]
    print(f"inner_size          : {inner_size}")


print(f"layer_norm_eps      : 1e-12 (RecBole default)")

print("\n--- Note on n_heads ---")
print("n_heads is NOT shape readable because query/key/value have shape")
print(f"  ({hidden_size}, {hidden_size}) regardless of n_heads.")
print("n_heads should be read from the HPO CSV or training log.")
print(f"CSV Test From 17: n_tests=4")
print(f"Verify: hidden_size={hidden_size} divisible for 4? {hidden_size % 4 == 0}")
print(f"Verify: hidden_size={hidden_size} divisible for 2? {hidden_size % 2 == 0}")
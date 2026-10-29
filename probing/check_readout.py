'''
Sanity check for the state probe readout. Run once per model before re-running the experiments:
    python check_readout.py <model_path>

1. Single token from a zero state: the state is exactly v k^T, so contracting the key axis
   ('hvk,hk->hv') must give back the direction of v (cos = 1); the old 'hkv,hk->hv' does not.
2. forward_seq (CUDA kernel) and step-by-step forward_one (python) must give the same state,
   so check 1 also holds for the state used in the experiments.
'''
# !!! set these before import RWKV !!!
import os
os.environ["RWKV_CUDA_ON"] = '1'

import sys
import torch
import torch.nn.functional as F
from rwkv_x.model_exp import RWKV_EXP
from rwkv_x.utils import PIPELINE

model = RWKV_EXP(model_path=sys.argv[1], strategy='cuda fp16')
tokenizer = PIPELINE(model).tokenizer
head_size = 64

def cos(a, b):
    return F.cosine_similarity(a, b, dim=-1).mean().item()

# 1. single token from zero state: S = v k^T
token = tokenizer.encode(" Einstein")[0]
_, state, k_list, v_list = model.forward([token], None)
n_layer = len(k_list)
worst_new, best_old = 1.0, -1.0
for i in range(n_layer):
    S = state[i * 3 + 1].float()                      # [H, V, K]
    H = S.shape[0]
    k = k_list[i].view(H, head_size).float()
    v = v_list[i].view(H, head_size).float()
    c_new = cos(torch.einsum('hvk,hk->hv', S, k), v)
    c_old = cos(torch.einsum('hkv,hk->hv', S, k), v)
    worst_new, best_old = min(worst_new, c_new), max(best_old, c_old)
    print(f"layer {i:02d} | cos new 'hvk' = {c_new:+.4f} | cos old 'hkv' = {c_old:+.4f}")
print(f"[check 1] min cos (new) = {worst_new:.4f}, max cos (old) = {best_old:.4f}")
assert worst_new > 0.99, "new readout should recover v exactly for a single write"

# 2. CUDA sequence path vs python step path
tokens = tokenizer.encode("The pass key is Einstein. The grass is green. The sky is blue.")
_, state_seq, _, _ = model.forward(tokens, None)
state_one = None
for t in tokens:
    _, state_one, _, _ = model.forward([t], state_one)
max_rel = 0.0
for i in range(n_layer):
    a, b = state_seq[i * 3 + 1].float(), state_one[i * 3 + 1].float()
    max_rel = max(max_rel, ((a - b).norm() / b.norm()).item())
print(f"[check 2] max relative state difference (seq vs one) = {max_rel:.2e}")
assert max_rel < 1e-2, "CUDA and python paths disagree on the state layout"
print("OK")

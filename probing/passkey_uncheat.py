'''
a script to probe the memory loss of a single value (one token)
Sequence length T, Distance is D, insert a value token at [T-D].
measure the memory loss of a single token at each layer, and how it changes with distance.
'''
# !!! set these before import RWKV !!!
from collections import defaultdict
from operator import gt, is_
import os
import token
os.environ["RWKV_CUDA_ON"] = '1'  # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

import sys
from tqdm import tqdm
import torch
import csv
from rwkv_x.model_exp import RWKV_EXP
from rwkv_x.utils import PIPELINE

VALUE_TOKEN_IDX = 54997 # Einstein

# download models: https://huggingface.co/BlinkDL/rwkv7-g1
model = RWKV_EXP(model_path=sys.argv[1], strategy='cuda fp16')
tokenizer = PIPELINE(model).tokenizer
from pathlib import Path
model_name = Path(sys.argv[1]).stem

information_line = f"The pass key is"
information_tokens = tokenizer.encode(information_line) # 4 tokens
passkey_tokens = information_tokens + [VALUE_TOKEN_IDX]
# =========================
# state probing
# =========================
def calculate_layer_memory_loss(state, k_list, v_list, p):
    """
    Args:
        state: model state
        k_list, v_list: list of length L
            each element shape: [T, D]

    Returns:
        layer_memory_loss: torch.Tensor, shape [L]
            per-layer memory loss for one sample
    """
    n_layer = len(k_list)
    T, n_embd = k_list[0].shape
    head_size = 64
    n_head = n_embd // head_size

    all_layer_memory_loss = []
    for i in range(n_layer):
        wkv_state = state[i * 3 + 1].float()   # [H, K, V], e.g. [16, 64, 64]
        k = k_list[i].view(T, n_head, head_size).float()   # [T, H, K]
        v = v_list[i].view(T, n_head, head_size).float()   # [T, H, V]

        # restore v from state and k
        # wkv_state: [H, K, V]
        # k:         [T, H, K]
        # output:    [T, H, V]
        v_restore = torch.einsum('hkv,thk->thv', wkv_state, k)

        # per-token, per-head mse -> then mean over head and token
        v_loss = ((v - v_restore) ** 2).mean(dim=-1).mean(dim=-1)   # [T]
        layer_p_loss = v_loss[p]  # scalar
        all_layer_memory_loss.append(layer_p_loss)

    all_layer_memory_loss = torch.stack(all_layer_memory_loss)  # [L]
    return all_layer_memory_loss


def compute_stats(values):
    """
    values: list[float]
    return: dict
    """
    x = torch.tensor(values, dtype=torch.float32)
    return {
        "count": x.numel(),
        "mean": x.mean().item(),
        "var": x.var(unbiased=False).item(),
        "min": x.min().item(),
        "max": x.max().item(),
        "median": x.median().item(),
    }


# =========================
# load dataset
# =========================
from datasets import load_dataset
ds = load_dataset("Jellyfish042/UncheatableEval-2026-01")['test']
select_idx = [i for i in range(0, len(ds), 15)] # select 500 samples to speed up
lines = [line['content'] for i, line in enumerate(ds) if i in select_idx]

# (dist, layer) -> list of losses
dist_layer_losses = defaultdict(list)

# raw records for csv
raw_records = []
D_list = [64, 256, 512, 1024, 2048] # distance from end
task = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
task_tokens = tokenizer.encode(task)
correct_count = defaultdict(int)
for sample_id, line in enumerate(tqdm(lines)):
    content = line
    content_tokens = tokenizer.encode(content)
    for D in D_list:
        prefix_tokens = content_tokens[:-D]
        suffix_tokens = content_tokens[-D:]
        # insert value token at position (T-D)
        tokens = task_tokens + prefix_tokens + passkey_tokens + suffix_tokens
        p = len(task_tokens) + len(prefix_tokens) + 4 # position of the value token
        out, state, k_list, v_list = model.forward(tokens, None)
        # num of layers [L]
        layer_memory_loss = calculate_layer_memory_loss(state, k_list, v_list, p)

        for layer_idx, loss in enumerate(layer_memory_loss):
            loss_val = loss.item()
            dist_layer_losses[(D, layer_idx)].append(loss_val)
            raw_records.append({
                "sample_id": sample_id,
                "layer": layer_idx,
                "distance": D,
                "memory_loss": loss_val,
            })
        # QA test
        final_question = "What is the pass key? The pass key is"
        question_tokens = tokenizer.encode(final_question)
        logits, state, _, _ = model(question_tokens, state)
        pred_token = torch.argmax(logits).item()
        gt_token = passkey_tokens[-1]
        is_correct = (pred_token == gt_token)
        # print(gt_token, pred_token, is_correct)
        if is_correct:
            correct_count[D] += 1

D2acc = {}
for D in correct_count:
    cnt = correct_count[D]
    acc = cnt / len(lines)
    D2acc[D] = acc
    print(f"Distance {D} | Accuracy: {acc:.4f}")
dist_acc_csv_path = model_name + ".dist_accuracy_uncheat.csv"
with open(dist_acc_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["distance", "accuracy"]
    )
    writer.writeheader()
    for D in sorted(D2acc.keys()):
        writer.writerow({
            "distance": D,
            "accuracy": D2acc[D]
        })
print(f"Saved distance-accuracy statistics to: {dist_acc_csv_path}")

# =========================
# print overall per-layer stats
# =========================
stats_records = []
all_layer_means = defaultdict(dict)

for (D, layer_idx) in sorted(dist_layer_losses.keys()):
    stats = compute_stats(dist_layer_losses[(D, layer_idx)])
    all_layer_means[D][layer_idx] = stats["mean"]

    record = {
        "layer": layer_idx,
        "distance": D,
        "count": stats["count"],
        "mean": stats["mean"],
        "var": stats["var"],
        "min": stats["min"],
        "max": stats["max"],
        "median": stats["median"],
    }
    stats_records.append(record)

print("\n" + "=" * 80)
for D in D_list:
    last_layer_index = max(all_layer_means[D].keys())
    last_layer = all_layer_means[D][last_layer_index]
    acc = D2acc[D]
    print(f"Distance {D} | Last Layer Memory Loss: {last_layer:.6f} | Accuracy: {acc:.4f}")
print("=" * 80)

# =========================
# save stats csv
# =========================
stats_csv_path = model_name + ".dist_layer_memory_loss_stats_uncheat.csv"
with open(stats_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["layer", "distance", "count", "mean", "var", "min", "max", "median"]
    )
    writer.writeheader()
    writer.writerows(stats_records)

print(f"Saved layer statistics to: {stats_csv_path}")


# =========================
# save raw csv
# =========================
raw_csv_path = model_name + ".dist_layer_memory_loss_raw_uncheat.csv"
with open(raw_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["sample_id", "layer", "distance", "memory_loss"]
    )
    writer.writeheader()
    writer.writerows(raw_records)

print(f"Saved raw per-sample per-layer losses to: {raw_csv_path}")
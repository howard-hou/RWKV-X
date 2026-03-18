'''
a script to probe the memory loss of a single value (one token)
Sequence length T, Distance is D, insert a value token at [T-D].
measure the memory loss of a single token at each layer, and how it changes with distance.
'''
# !!! set these before import RWKV !!!
from collections import defaultdict
import os
os.environ["RWKV_CUDA_ON"] = '1'  # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

import sys
from tqdm import tqdm
import torch
import numpy as np
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

def filter_docs_by_length(lines, tokenizer, d_list, n_docs=500):
    doc_lens = [ len(tokenizer.encode(line)) for line in lines ]
    d2docs = {}
    for D in d_list:
        filtered_docs = []
        for i in range(len(lines)):
            doc = lines[i]
            doc_len = doc_lens[i]
            if doc_len >= D:
                filtered_docs.append(doc)
        if len(filtered_docs) > n_docs:
            step = len(filtered_docs) // n_docs
            filtered_docs = filtered_docs[::step][:n_docs]
        else:
            filtered_docs = filtered_docs[:n_docs]
        d2docs[D] = filtered_docs
    for D in d_list:
        print(f"Token Lag {D} | Found {len(d2docs[D])} documents with length >= {D}")
    return d2docs
# =========================
# load dataset
# =========================
D_list = [64, 256, 512, 1024, 2048] # distance from end
from datasets import load_dataset
ds = load_dataset("Jellyfish042/UncheatableEval-2026-01-Long")['test']
lines = [line['content'] for line in ds]
# set docs for D
d2docs = filter_docs_by_length(lines, tokenizer, D_list, n_docs=500)
# (dist, layer) -> list of losses
dist_layer_losses = defaultdict(list)

# raw records for csv
raw_records = []
task = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
task_tokens = tokenizer.encode(task)
correct_count = defaultdict(int)
loss_vs_binary = [] # (loss, is_correct)
for D in tqdm(D_list):
    docs = d2docs[D]
    for sample_id, line in enumerate(docs):
        content = line
        content_tokens = tokenizer.encode(content)
        prefix_tokens = content_tokens[:-D]
        suffix_tokens = content_tokens[-D:]
        # insert value token at position (T-D)
        tokens = task_tokens + prefix_tokens + passkey_tokens + suffix_tokens
        p = len(task_tokens) + len(prefix_tokens) + 4 # position of the value token
        with torch.inference_mode():
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
        mean_loss = layer_memory_loss.mean().item()
        # QA test
        final_question = "What is the pass key? The pass key is"
        question_tokens = tokenizer.encode(final_question)
        with torch.inference_mode():
            logits, state, _, _ = model(question_tokens, state)
        pred_token = torch.argmax(logits).item()
        gt_token = passkey_tokens[-1]
        is_correct = (pred_token == gt_token)
        # print(gt_token, pred_token, is_correct)
        if is_correct:
            correct_count[D] += 1
            loss_vs_binary.append((mean_loss, 1))
        else:
            loss_vs_binary.append((mean_loss, 0))

D2acc = {}
for D in D_list:
    cnt = correct_count[D] if D in correct_count else 0
    acc = cnt / len(d2docs[D])
    D2acc[D] = acc
    print(f"Token Lag {D} | Accuracy: {acc:.4f}")
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
    # # max loss layer index
    # max_layer_index = max(all_layer_means[D].keys(), key=lambda idx: all_layer_means[D][idx])
    # max_layer = all_layer_means[D][max_layer_index]
    mean_loss = np.mean([ all_layer_means[D][idx] for idx in all_layer_means[D] ])
    acc = D2acc[D]
    print(f"Token Lag {D} | Mean Memory Loss: {mean_loss:.6f} | Accuracy: {acc:.4f}")
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

# =========================
# save loss vs binary csv
# =========================
loss_binary_csv_path = model_name + ".loss_vs_binary_uncheat.csv"
with open(loss_binary_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["mean_loss", "is_correct"]
    )
    writer.writeheader()
    for mean_loss, is_correct in loss_vs_binary:
        writer.writerow({
            "mean_loss": mean_loss,
            "is_correct": is_correct
        })
print(f"Saved loss vs binary statistics to: {loss_binary_csv_path}")
# plot loss vs binary, x is loss, y is binary
import matplotlib.pyplot as plt
import seaborn as sns
sns.violinplot(x=[ x for x, y in loss_vs_binary ], y=[ y for x, y in loss_vs_binary ])
plt.title("Memory Loss vs QA Accuracy")
plt.xlabel("Mean Memory Loss")
plt.ylabel("QA Accuracy (0 or 1)")
plt.savefig(model_name + ".loss_vs_binary_uncheat.png")
print(f"Saved loss vs binary plot to: {model_name + '.loss_vs_binary_uncheat.png'}")
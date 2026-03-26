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
TOPK_LIST = [128, 256, 512, 1024]

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
def calculate_layer_memory_loss(state, k_list, v_list):
    """
    Args:
        state: model state
        k_list, v_list: list of length L
            each element shape: [T, D]

    Returns:
        layer_memory_loss: torch.Tensor, shape [L, T]
            per-layer, per-token memory loss for one sample
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
        all_layer_memory_loss.append(v_loss)

    all_layer_memory_loss = torch.stack(all_layer_memory_loss)  # [L, T]
    return all_layer_memory_loss


def calculate_topk_hits(token_loss, p, topk_list):
    topk2hit = {}
    n_tokens = token_loss.shape[0]
    for topk in topk_list:
        k = min(topk, n_tokens)
        topk_indices = torch.topk(token_loss, k=k, largest=True).indices
        topk2hit[topk] = int((topk_indices == p).any().item())
    return topk2hit


def calculate_avg_layer_topk_hits(all_layer_memory_loss, p, topk_list):
    avg_layer_loss = all_layer_memory_loss.mean(dim=0)  # [T]
    return calculate_topk_hits(avg_layer_loss, p, topk_list)


def calculate_per_layer_topk_hits(all_layer_memory_loss, p, topk_list):
    layer_topk_hits = []
    for layer_loss in all_layer_memory_loss:
        layer_topk_hits.append(calculate_topk_hits(layer_loss, p, topk_list))
    return layer_topk_hits


def filter_docs_by_length(lines, tokenizer, d_list, n_docs=500):
    doc_lens = [ len(tokenizer.encode(line)) for line in lines ]
    d2docs = {}
    d2lens = {}
    for D in d_list:
        filtered_docs = []
        filtered_doc_lens = []
        for i in range(len(lines)):
            doc = lines[i]
            doc_len = doc_lens[i]
            if doc_len >= D:
                filtered_docs.append(doc)
                filtered_doc_lens.append(doc_len)
        if len(filtered_docs) > n_docs:
            step = len(filtered_docs) // n_docs
            filtered_docs = filtered_docs[::step][:n_docs]
            filtered_doc_lens = filtered_doc_lens[::step][:n_docs]
        else:
            filtered_docs = filtered_docs[:n_docs]
            filtered_doc_lens = filtered_doc_lens[:n_docs]
        d2docs[D] = filtered_docs
        d2lens[D] = filtered_doc_lens
    for D in d_list:
        print(f"Token Lag {D} | Found {len(d2docs[D])} documents with length >= {D}")
    return d2docs, d2lens
# =========================
# load dataset
# =========================
D_list = [2048, 4096, 8192] # distance from end
from datasets import load_dataset
ds = load_dataset("Jellyfish042/UncheatableEval-2026-01-Long")['test']
lines = [line['content'] for line in ds]
# set docs for D
d2docs, d2lens = filter_docs_by_length(lines, tokenizer, D_list, n_docs=500)

task = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
task_tokens = tokenizer.encode(task)
avg_hit_counts = defaultdict(int)
layer_hit_counts = defaultdict(int)
sample_counts = defaultdict(int)
n_layer = None

for D in tqdm(D_list):
    docs = d2docs[D]
    for line in docs:
        content = line
        content_tokens = tokenizer.encode(content)
        prefix_tokens = content_tokens[:-D]
        suffix_tokens = content_tokens[-D:]
        # insert value token at position (T-D)
        tokens = task_tokens + prefix_tokens + passkey_tokens + suffix_tokens
        p = len(task_tokens) + len(prefix_tokens) + 4 # position of the value token
        with torch.inference_mode():
            _, state, k_list, v_list = model.forward(tokens, None)

        layer_memory_loss = calculate_layer_memory_loss(state, k_list, v_list)
        if n_layer is None:
            n_layer = layer_memory_loss.shape[0]

        avg_topk2hit = calculate_avg_layer_topk_hits(layer_memory_loss, p, TOPK_LIST)
        layer_topk2hit = calculate_per_layer_topk_hits(layer_memory_loss, p, TOPK_LIST)

        sample_counts[D] += 1
        for topk, hit in avg_topk2hit.items():
            avg_hit_counts[(D, topk)] += hit

        for layer_idx, topk2hit in enumerate(layer_topk2hit):
            for topk, hit in topk2hit.items():
                layer_hit_counts[(D, layer_idx, topk)] += hit

# =========================
# save averaged-layer hitrate csv
# =========================
avg_csv_path = model_name + ".token_lag_topk_hitrate_avg_uncheat.csv"
with open(avg_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["avg_doc_len", "token_lag", "topk", "hit", "total_samples", "hit_rate"]
    )
    writer.writeheader()
    for D in D_list:
        total_samples = sample_counts[D]
        avg_doc_len = np.mean(d2lens[D])
        for topk in TOPK_LIST:
            hit = avg_hit_counts[(D, topk)]
            writer.writerow({
                "avg_doc_len": avg_doc_len,
                "token_lag": D,
                "topk": topk,
                "hit": hit,
                "total_samples": total_samples,
                "hit_rate": (hit / total_samples) if total_samples > 0 else 0.0,
            })

print(f"Saved averaged-layer topk hitrate statistics to: {avg_csv_path}")

# =========================
# save per-layer hitrate csv
# =========================
layer_csv_path = model_name + ".token_lag_topk_hitrate_per_layer_uncheat.csv"
with open(layer_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["avg_doc_len", "token_lag", "layer", "topk", "hit", "total_samples", "hit_rate"]
    )
    writer.writeheader()
    if n_layer is not None:
        for D in D_list:
            total_samples = sample_counts[D]
            avg_doc_len = np.mean(d2lens[D])
            for layer_idx in range(n_layer):
                for topk in TOPK_LIST:
                    hit = layer_hit_counts[(D, layer_idx, topk)]
                    writer.writerow({
                        "avg_doc_len": avg_doc_len,
                        "token_lag": D,
                        "layer": layer_idx,
                        "topk": topk,
                        "hit": hit,
                        "total_samples": total_samples,
                        "hit_rate": (hit / total_samples) if total_samples > 0 else 0.0,
                    })

print(f"Saved per-layer topk hitrate statistics to: {layer_csv_path}")

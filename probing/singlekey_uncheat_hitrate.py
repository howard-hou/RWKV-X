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
TOPK_LIST = [4, 16, 64, 256]

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
        all_layer_memory_loss.append(v_loss)

    all_layer_memory_loss = torch.stack(all_layer_memory_loss)  # [L, T]
    return all_layer_memory_loss


def calculate_hit_rate_in_topk(all_layer_memory_loss, p, topk_list):
    avg_layer_loss = all_layer_memory_loss.mean(dim=0)  # [T]
    topk2hit = {}
    for topk in topk_list:
        topk_indices = torch.topk(avg_layer_loss, k=topk, largest=True).indices
        topk2hit[topk] = int((topk_indices == p).any().item())

    return topk2hit

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
all_topk2hit = defaultdict(int)
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
        layer_memory_loss = calculate_layer_memory_loss(state, k_list, v_list)
        topk2hit = calculate_hit_rate_in_topk(layer_memory_loss, p, TOPK_LIST)
        for topk in topk2hit:
            

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
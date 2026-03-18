'''
a script to probe the memory loss of a single value (one token)
Sequence length T, Distance is D, insert a value token at [T-D].
measure the memory loss of a single token at each layer, and how it changes with distance.
'''
# !!! set these before import RWKV !!!
from collections import defaultdict
from operator import gt, is_
import os
os.environ["RWKV_CUDA_ON"] = '1'  # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

import sys
from tqdm import tqdm
import torch
import numpy as np
import csv
from rwkv_x.model_exp import RWKV_EXP
from rwkv_x.utils import PIPELINE

# VALUE_TOKEN_IDX = 54997 # Einstein
VALUE_TOKEN_LIST = list(range(0, 65529)) #
VALUE_TOKEN_LIST = list(range(0, 5)) #

# download models: https://huggingface.co/BlinkDL/rwkv7-g1
model = RWKV_EXP(model_path=sys.argv[1], strategy='cuda fp16')
tokenizer = PIPELINE(model).tokenizer
from pathlib import Path
model_name = Path(sys.argv[1]).stem

information_line = f"The pass key is"
information_tokens = tokenizer.encode(information_line) #
all_passkey_tokens = []
for idx in VALUE_TOKEN_LIST:
    passkey_tokens = information_tokens + [idx]
    all_passkey_tokens.append(passkey_tokens)
# =========================
# state probing
# =========================
def calculate_mean_memory_loss(state, k_list, v_list, p):
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
    return all_layer_memory_loss.mean()


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
# (dist, layer) -> list of losses
dist_layer_losses = defaultdict(list)

# raw records for csv
token2record = {}
D_list = [d for d in range(1024, 1024*10+1, 1024)]
garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
garbage_tokens = tokenizer.encode(garbage) # 24 tokens
prefix_tokens = garbage_tokens * 5 # 120 tokens
suffix_tokens = garbage_tokens * 1000 # 24000 tokens
correct_count = defaultdict(int)
for passkey_tokens in tqdm(all_passkey_tokens):
    gt_token = passkey_tokens[-1]
    record = {"token_lag": [], "memory_loss": [], "is_correct": []}
    for D in D_list:
        # insert value token at position (T-D)
        tokens = prefix_tokens + passkey_tokens + suffix_tokens[:D]
        p = len(prefix_tokens) + 4 # position of the value token
        out, state, k_list, v_list = model.forward(tokens, None)
        # num of layers [L]
        memory_loss = calculate_mean_memory_loss(state, k_list, v_list, p)
        loss_val = memory_loss.item()
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
        record["token_lag"].append(D)
        record["memory_loss"].append(loss_val)
        record["is_correct"].append(is_correct)
    # update
    token2record[gt_token] = record

# process records
for token in token2record:
    text = tokenizer.decode([token])
    record = token2record[token]
    record["accuracy"] = np.mean(record["is_correct"])
    record["mean_loss"] = np.mean(record["memory_loss"])
    # max retraining distance, the first is_correct becomes False
    if all(record["is_correct"]):
        record["max_retrain_lag"] = D_list[-1]
    elif all([not x for x in record["is_correct"]]):
        record["max_retrain_lag"] = 0
    else:
        for i, is_correct in enumerate(record["is_correct"]):
            if not is_correct:
                record["max_retrain_lag"] = record["token_lag"][i-1]
                break
    record["text"] = text
# save to csv
csv_path = f"{model_name}.token_forgetting_analysis.csv"
with open(csv_path, "w", newline='') as csvfile:
    fieldnames = ["token", "accuracy", "memory_loss", "max_retrain_lag"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for token, record in token2record.items():
        row = {
            "token": token,
            "accuracy": record["accuracy"],
            "memory_loss": round(record["mean_loss"], 3),
            "max_retrain_lag": record["max_retrain_lag"],
        }
        writer.writerow(row)
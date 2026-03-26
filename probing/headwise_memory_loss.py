# !!! set these before import RWKV !!!
from collections import defaultdict
import os
os.environ["RWKV_CUDA_ON"] = '1'  # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

import sys
from tqdm import tqdm
import torch
import csv
from pathlib import Path
from datasets import load_dataset
from rwkv_x.model_exp import RWKV_EXP
from rwkv_x.utils import PIPELINE


# download models: https://huggingface.co/BlinkDL/rwkv7-g1
model = RWKV_EXP(model_path=sys.argv[1], strategy='cuda fp16')
tokenizer = PIPELINE(model).tokenizer
from pathlib import Path
model_name = Path(sys.argv[1]).stem

# =========================
# state probing
# =========================
def calculate_headwise_memory_loss(state, k_list, v_list):
    """
    Args:
        state: model state
        k_list, v_list: list of length L
            each element shape: [T, D]

    Returns:
        headwise_memory_loss: torch.Tensor, shape [L, H]
            per-layer, per-head memory loss for one sample
    """
    n_layer = len(k_list)
    seq_len, n_embd = k_list[0].shape
    head_size = 64
    n_head = n_embd // head_size

    all_layer_head_losses = []
    for i in range(n_layer):
        wkv_state = state[i * 3 + 1].float()   # [H, K, V], e.g. [16, 64, 64]
        k = k_list[i].view(seq_len, n_head, head_size).float()   # [T, H, K]
        v = v_list[i].view(seq_len, n_head, head_size).float()   # [T, H, V]

        # restore v from state and k
        # wkv_state: [H, K, V]
        # k:         [T, H, K]
        # output:    [T, H, V]
        v_restore = torch.einsum('hkv,thk->thv', wkv_state, k)

        # keep head dimension, average only over head_size and token dims
        head_loss = ((v - v_restore) ** 2).mean(dim=-1).mean(dim=0)   # [H]
        all_layer_head_losses.append(head_loss)

    all_layer_head_losses = torch.stack(all_layer_head_losses)  # [L, H]
    return all_layer_head_losses


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
ds = load_dataset("Jellyfish042/UncheatableEval-2026-01")["test"]

# (layer, head) -> list of loss
overall_head_losses = defaultdict(list)

# raw records for csv
raw_records = []

for sample_id, line in enumerate(tqdm(ds)):
    content = line["content"]
    tokens = tokenizer.encode(content)

    _, state, k_list, v_list = model.forward(tokens, None)

    # [L, H]
    headwise_memory_loss = calculate_headwise_memory_loss(state, k_list, v_list)

    for layer_idx, layer_losses in enumerate(headwise_memory_loss):
        for head_idx, loss in enumerate(layer_losses):
            loss_val = loss.item()
            overall_head_losses[(layer_idx, head_idx)].append(loss_val)
            raw_records.append({
                "sample_id": sample_id,
                "layer": layer_idx,
                "head": head_idx,
                "memory_loss": loss_val,
            })


# =========================
# print overall per-layer per-head stats
# =========================
print("\n" + "=" * 80)
print("Overall Per-Layer Per-Head Memory Loss Statistics")
print("=" * 80)

stats_records = []
all_head_means = []

for layer_idx, head_idx in sorted(overall_head_losses.keys()):
    stats = compute_stats(overall_head_losses[(layer_idx, head_idx)])
    all_head_means.append(stats["mean"])

    record = {
        "layer": layer_idx,
        "head": head_idx,
        "count": stats["count"],
        "mean": stats["mean"],
        "var": stats["var"],
        "min": stats["min"],
        "max": stats["max"],
        "median": stats["median"],
    }
    stats_records.append(record)

    print(
        f"Layer {layer_idx:02d} Head {head_idx:02d} | "
        f"count={record['count']:4d} | "
        f"mean={record['mean']:.6f} | "
        f"var={record['var']:.6f} | "
        f"min={record['min']:.6f} | "
        f"max={record['max']:.6f} | "
        f"median={record['median']:.6f}"
    )

overall_avg = sum(all_head_means) / len(all_head_means)
print("\n" + "=" * 80)
print(f"Overall Average Across Layer-Head Pairs (mean of pair means): {overall_avg:.6f}")
print("=" * 80)


# =========================
# save stats csv
# =========================
stats_csv_path = model_name + ".headwise_memory_loss_stats.csv"
with open(stats_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["layer", "head", "count", "mean", "var", "min", "max", "median"]
    )
    writer.writeheader()
    writer.writerows(stats_records)

print(f"Saved layer-head statistics to: {stats_csv_path}")


# =========================
# save raw csv
# =========================
model_name = Path(sys.argv[1]).stem
raw_csv_path = model_name + ".headwise_memory_loss_raw.csv"
with open(raw_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["sample_id", "layer", "head", "memory_loss"]
    )
    writer.writeheader()
    writer.writerows(raw_records)

print(f"Saved raw per-sample per-layer per-head losses to: {raw_csv_path}")

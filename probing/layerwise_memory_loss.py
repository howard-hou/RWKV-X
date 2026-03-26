# !!! set these before import RWKV !!!
from collections import defaultdict
import os
os.environ["RWKV_CUDA_ON"] = '1'  # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

import sys
from tqdm import tqdm
import torch
import csv
from rwkv_x.model_exp import RWKV_EXP
from rwkv_x.utils import PIPELINE

T = 1024

# download models: https://huggingface.co/BlinkDL/rwkv7-g1
model = RWKV_EXP(model_path=sys.argv[1], strategy='cuda fp16')
tokenizer = PIPELINE(model).tokenizer


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
        layer_loss = v_loss.mean()   # scalar
        all_layer_memory_loss.append(layer_loss)

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

# layer -> list of loss
overall_layer_losses = defaultdict(list)

# raw records for csv
raw_records = []

for sample_id, line in enumerate(tqdm(ds)):
    content = line['content']
    tokens = tokenizer.encode(content)

    out, state, k_list, v_list = model.forward(tokens, None)

    # [L]
    layer_memory_loss = calculate_layer_memory_loss(state, k_list, v_list)

    for layer_idx, loss in enumerate(layer_memory_loss):
        loss_val = loss.item()
        overall_layer_losses[layer_idx].append(loss_val)
        raw_records.append({
            "sample_id": sample_id,
            "layer": layer_idx,
            "memory_loss": loss_val,
        })


# =========================
# print overall per-layer stats
# =========================
print("\n" + "=" * 80)
print("Overall Per-Layer Memory Loss Statistics")
print("=" * 80)

stats_records = []
all_layer_means = []

for layer_idx in sorted(overall_layer_losses.keys()):
    stats = compute_stats(overall_layer_losses[layer_idx])
    all_layer_means.append(stats["mean"])

    record = {
        "layer": layer_idx,
        "count": stats["count"],
        "mean": stats["mean"],
        "var": stats["var"],
        "min": stats["min"],
        "max": stats["max"],
        "median": stats["median"],
    }
    stats_records.append(record)

    print(
        f"Layer {layer_idx:02d} | "
        f"count={record['count']:4d} | "
        f"mean={record['mean']:.6f} | "
        f"var={record['var']:.6f} | "
        f"min={record['min']:.6f} | "
        f"max={record['max']:.6f} | "
        f"median={record['median']:.6f}"
    )

overall_avg = sum(all_layer_means) / len(all_layer_means)
print("\n" + "=" * 80)
print(f"Overall Average Across Layers (mean of layer means): {overall_avg:.6f}")
print("=" * 80)


# =========================
# save stats csv
# =========================
stats_csv_path = "layer_memory_loss_stats.csv"
with open(stats_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["layer", "count", "mean", "var", "min", "max", "median"]
    )
    writer.writeheader()
    writer.writerows(stats_records)

print(f"Saved layer statistics to: {stats_csv_path}")


# =========================
# save raw csv
# =========================
from pathlib import Path
model_name = Path(sys.argv[1]).stem
raw_csv_path = model_name + "-layer_memory_loss_raw.csv"
with open(raw_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["sample_id", "layer", "memory_loss"]
    )
    writer.writeheader()
    writer.writerows(raw_records)

print(f"Saved raw per-sample per-layer losses to: {raw_csv_path}")
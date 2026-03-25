########################################################################################################
# The RWKV-X Language Model - https://github.com/howard-hou/RWKV-X
########################################################################################################
#
# pip install rwkv lm_eval --upgrade
# previous version only support lm_eval==0.3.0
# this version support lm_eval>=0.4.0
#
import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

np.set_printoptions(precision=4, suppress=True, linewidth=200)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

os.environ["RWKV_JIT_ON"] = "0"
os.environ["RWKV_CUDA_ON"] = "1"
os.environ["RWKV_V7_ON"] = "1"

from rwkv_x.model import RWKV_X, RWKV_X_Config


DEFAULT_MAX_SEQ_LENGTHS = [1000, 2000, 4000, 8000]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark RWKV-X prefill and decoding latency.")
    parser.add_argument("model_path", type=str)
    parser.add_argument("--log_dir", type=Path, default=Path("logs/decoding"))
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32", "bf16"])
    parser.add_argument("--head_size", type=int, default=64, help="RWKV_X_Config.head_size")
    parser.add_argument("--chunk_len", type=int, default=4000, help="Prefill chunk length")
    parser.add_argument("--decode_steps", type=int, default=10, help="Number of decode tokens per benchmark")
    parser.add_argument(
        "--max_seq_lengths",
        type=int,
        nargs="+",
        default=DEFAULT_MAX_SEQ_LENGTHS,
        help="Context lengths to benchmark",
    )

    args = parser.parse_args()

    if args.head_size <= 0:
        parser.error("--head_size must be a positive integer")
    if args.chunk_len <= 0:
        parser.error("--chunk_len must be a positive integer")
    if args.decode_steps <= 0:
        parser.error("--decode_steps must be a positive integer")
    if any(length < 0 for length in args.max_seq_lengths):
        parser.error("--max_seq_lengths only accepts non-negative integers")
    if args.device.startswith("cpu") and args.dtype == "fp16":
        parser.error("fp16 is not supported on CPU. Use --dtype fp32 or --dtype bf16 instead.")

    return args


def is_cuda_device(device: str) -> bool:
    return device.startswith("cuda")


def configure_device(device: str):
    if not is_cuda_device(device):
        return
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device} was requested, but torch.cuda.is_available() is False.")
    torch.cuda.set_device(device)


def reset_peak_memory(device: str):
    if not is_cuda_device(device):
        return
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def get_peak_memory_gib(device: str) -> float:
    if not is_cuda_device(device):
        return float("nan")
    return torch.cuda.max_memory_allocated(device) / 1024 ** 3


def format_memory_gib(memory_gib: float) -> str:
    if math.isnan(memory_gib):
        return "N/A"
    return f"{memory_gib:.2f} GiB"


def sanitize_tag(value: str) -> str:
    return value.replace(":", "_").replace("/", "_").replace("\\", "_")


args = parse_args()
model_name = args.model_path
model_stem = Path(model_name).stem
output_dir = args.log_dir
output_dir.mkdir(parents=True, exist_ok=True)

strategy = f"{args.device} {args.dtype}"
config = RWKV_X_Config(head_size=args.head_size)

print(f"Loading model - {model_name}")
print(f"Strategy: {strategy}")
print(f"Config: {config}")

configure_device(args.device)
model = RWKV_X(model_path=args.model_path, strategy=strategy, config=config)

print(f"Model loaded on {args.device}")
print(f"Context length test: {args.max_seq_lengths}")
print(f"Prefill chunk length: {args.chunk_len}")
print(f"Decode steps: {args.decode_steps}")

prefill_records = []
decoding_records = []

for ctx_len in args.max_seq_lengths:
    tokens = [0] * ctx_len

    reset_peak_memory(args.device)
    state = None
    start_prefill = time.time()
    remaining_tokens = tokens
    while remaining_tokens:
        _, state = model.forward(remaining_tokens[: args.chunk_len], state)
        remaining_tokens = remaining_tokens[args.chunk_len :]
    end_prefill = time.time()
    prefill_latency = (end_prefill - start_prefill) * 1000
    prefill_mem = get_peak_memory_gib(args.device)
    print(
        f"[PREFILL] ctx_len: {ctx_len}, latency: {prefill_latency:.2f} ms, "
        f"memory: {format_memory_gib(prefill_mem)}"
    )
    prefill_records.append(
        {
            "ctx_len": ctx_len,
            "latency": round(prefill_latency, 2),
            "memory": round(prefill_mem, 2) if not math.isnan(prefill_mem) else np.nan,
        }
    )

    reset_peak_memory(args.device)
    start_decode = time.time()
    for token_id in range(args.decode_steps):
        _, state = model.forward([token_id], state)
    end_decode = time.time()
    decoding_latency = (end_decode - start_decode) / args.decode_steps * 1000
    decoding_mem = get_peak_memory_gib(args.device)
    print(
        f"[DECODING] ctx_len: {ctx_len}, latency: {decoding_latency:.2f} ms, "
        f"memory: {format_memory_gib(decoding_mem)}"
    )
    decoding_records.append(
        {
            "ctx_len": ctx_len,
            "latency": round(decoding_latency, 2),
            "memory": round(decoding_mem, 2) if not math.isnan(decoding_mem) else np.nan,
        }
    )

prefill_df = pd.DataFrame(prefill_records)
decoding_df = pd.DataFrame(decoding_records)
combined_df = prefill_df.merge(decoding_df, on="ctx_len", suffixes=("_prefill", "_decoding"))

output_name = (
    f"{model_stem}_device-{sanitize_tag(args.device)}_dtype-{args.dtype}"
    f"_hs{args.head_size}_chunk{args.chunk_len}.csv"
)
output_path = output_dir / output_name
combined_df.to_csv(output_path, index=False)
print(f"Saved benchmark results to {output_path}")

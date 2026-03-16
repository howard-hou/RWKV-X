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
import random
from tqdm import tqdm
import torch
import csv
from rwkv_x.model_exp import RWKV_EXP
from rwkv_x.utils import PIPELINE

VALUE_TOKEN_IDX = 54997 # Einstein
DISTRACTOR_TOKEN_IDXS = [32941, 15503, 41820]  # replace with real ids

# download models: https://huggingface.co/BlinkDL/rwkv7-g1
model = RWKV_EXP(model_path=sys.argv[1], strategy='cuda fp16')
tokenizer = PIPELINE(model).tokenizer
from pathlib import Path
model_name = Path(sys.argv[1]).stem

true_information_line = "The final pass key is"
true_information_tokens = tokenizer.encode(true_information_line)
true_passkey_tokens = true_information_tokens + [VALUE_TOKEN_IDX]

distractor_information_line = "The old pass key is"
distractor_information_tokens = tokenizer.encode(distractor_information_line)
# =========================
# build context with distractors
# =========================
def build_distractor_passkey_context(
    suffix_tokens,
    distractor_token_idxs,
    distractor_information_tokens,
):
    '''
    insert distractor tokens after the true passkey tokens
    '''
    new_suffix_tokens = []
    random_pos_list = [random.randrange(len(suffix_tokens) + 1) for _ in distractor_token_idxs]
    random_pos_list.sort() # ensure the order of distractor blocks is the same across samples, to reduce variance
    for tid in distractor_token_idxs:
        distractor_tokens = distractor_information_tokens + [tid]
        # select random position to insert distractor blocks
        random_pos = random_pos_list.pop(0)
        new_suffix_tokens.extend(suffix_tokens[:random_pos] + distractor_tokens)
        suffix_tokens = suffix_tokens[random_pos:]
    return new_suffix_tokens

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

def merge_docs_until_exceed_max_tokens(lines, tokenizer, max_D, sep="\n\n"):
    """
    Greedily merge consecutive documents until tokenized length > max_D.

    Rules:
    1. If a single document already has token length > max_D, keep it as is.
    2. If a single document has token length <= max_D, keep merging the next
       document(s) until the merged token length > max_D.
    3. Process documents sequentially without overlap.

    Args:
        lines (list[str]): each element is an original document
        tokenizer: HuggingFace tokenizer or compatible tokenizer
        max_D (int): token-length threshold
        sep (str): separator between merged documents

    Returns:
        merged_docs (list[str]): merged document texts
        merged_indices (list[list[int]]): original indices used in each merged doc
        merged_token_lens (list[int]): token lengths of merged docs
    """
    def token_len(text):
        return len(tokenizer.encode(text))

    merged_docs = []
    merged_indices = []
    merged_token_lens = []

    i = 0
    n = len(lines)

    while i < n:
        cur_text = lines[i]
        cur_indices = [i]
        cur_len = token_len(cur_text)

        # single document already exceeds max_D, keep it as is
        if cur_len > max_D:
            merged_docs.append(cur_text)
            merged_indices.append(cur_indices)
            merged_token_lens.append(cur_len)
            i += 1
            continue

        # greedily merge next documents until exceeding max_D
        j = i + 1
        while j < n and cur_len <= max_D:
            cur_text = cur_text + sep + lines[j]
            cur_indices.append(j)
            cur_len = token_len(cur_text)
            j += 1

        # 
        if cur_len >= max_D:
            merged_docs.append(cur_text)
            merged_indices.append(cur_indices)
            merged_token_lens.append(cur_len)

        i = j

    return merged_docs, merged_indices, merged_token_lens
# =========================
# load dataset
# =========================
D_list = [64, 256, 512, 1024, 2048] # distance from end
max_D = max(D_list)
from datasets import load_dataset
ds = load_dataset("Jellyfish042/UncheatableEval-2026-01")['test']
lines = [line['content'] for line in ds]
merged_docs, merged_indices, merged_token_lens = merge_docs_until_exceed_max_tokens(lines, tokenizer, max_D)
num_merged_docs = len(merged_docs)
print(f"Total original samples: {len(lines)}")
print(f"Total merged samples: {num_merged_docs}")
if num_merged_docs > 500: # select 500 samples to speed up
    step = num_merged_docs // 500
    select_idx = [i for i in range(0, num_merged_docs, step)] # select 500 samples to speed up
    docs = [merged_docs[i] for i in select_idx][:500] # ensure at most 500 samples
else:
    docs = merged_docs
# docs = docs[:10]
print(f"Selected {len(docs)} samples for probing.")
# (dist, layer) -> list of losses
dist_layer_losses = defaultdict(list)

# raw records for csv
raw_records = []
task = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
task_tokens = tokenizer.encode(task)
correct_count = defaultdict(int)
for sample_id, line in enumerate(tqdm(docs)):
    content = line
    content_tokens = tokenizer.encode(content)
    for D in D_list:
        real_D = D - len(DISTRACTOR_TOKEN_IDXS) * (len(distractor_information_tokens) + 1) # real distance after inserting distractors
        prefix_tokens = content_tokens[:-real_D]
        suffix_tokens = content_tokens[-real_D:]
        suffix_tokens = build_distractor_passkey_context(suffix_tokens, distractor_information_tokens, DISTRACTOR_TOKEN_IDXS)
        # insert value token at position (T-D)
        tokens = task_tokens + prefix_tokens + true_passkey_tokens + suffix_tokens
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
        final_question = "What is the final pass key? The final pass key is"
        question_tokens = tokenizer.encode(final_question)
        logits, state, _, _ = model(question_tokens, state)
        pred_token = torch.argmax(logits).item()
        gt_token = true_passkey_tokens[-1]
        is_correct = (pred_token == gt_token)
        # print(gt_token, pred_token, is_correct)
        if is_correct:
            correct_count[D] += 1

D2acc = {}
for D in correct_count:
    cnt = correct_count[D]
    acc = cnt / len(docs)
    D2acc[D] = acc
    print(f"Distance {D} | Accuracy: {acc:.4f}")
dist_acc_csv_path = model_name + ".dist_accuracy_distract.csv"
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
    # max loss layer index
    max_layer_index = max(all_layer_means[D].keys(), key=lambda idx: all_layer_means[D][idx])
    max_layer = all_layer_means[D][max_layer_index]
    if D in D2acc:
        acc = D2acc[D]
        print(f"Distance {D} | Max Memory Loss: {max_layer:.6f} | Accuracy: {acc:.4f}")
print("=" * 80)

# =========================
# save stats csv
# =========================
stats_csv_path = model_name + ".dist_layer_memory_loss_stats_distract.csv"
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
raw_csv_path = model_name + ".dist_layer_memory_loss_raw_distract.csv"
with open(raw_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["sample_id", "layer", "distance", "memory_loss"]
    )
    writer.writeheader()
    writer.writerows(raw_records)

print(f"Saved raw per-sample per-layer losses to: {raw_csv_path}")
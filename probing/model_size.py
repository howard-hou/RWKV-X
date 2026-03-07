# !!! set these before import RWKV !!!
from collections import defaultdict
import os
os.environ["RWKV_CUDA_ON"] = '1' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries
import sys
from tqdm import tqdm
import torch
import torch.nn.functional as F
from rwkv_x.model_exp import RWKV_EXP
from rwkv_x.utils import PIPELINE

T = 1024
# download models: https://huggingface.co/BlinkDL/rwkv7-g1
model = RWKV_EXP(model_path=sys.argv[1], strategy='cuda fp16')
tokenizer = PIPELINE(model).tokenizer

# state probing
def calculate_memory_loss(state, k_list, v_list):
    # state: [D]
    # k_list, v_list: [L, T, D]
    # calculate the loss between state and the average of k_list and v_list
    n_layer = len(k_list)
    T, n_embd = k_list[0].shape
    head_size = 64
    n_head = n_embd // head_size
    all_layer_memory_loss = []
    for i in range(n_layer):
        wkv_state = state[i*3+1]
        k = k_list[i].view(T, n_head, head_size).float()
        v = v_list[i].view(T, n_head, head_size).float()
        v_restore = torch.einsum('imn,tin->tim', wkv_state, k)
        v_loss = ((v - v_restore) ** 2).mean(-1).mean(-1)
        all_layer_memory_loss.append(v_loss)
    all_layer_memory_loss = torch.stack(all_layer_memory_loss)
    return all_layer_memory_loss.mean()

# load dataset, use uncheatable eval
from datasets import load_dataset
ds = load_dataset("Jellyfish042/UncheatableEval-2026-01")['test']
category2loss = defaultdict(list)
for line in tqdm(ds):
    content = line['content']
    tokens = tokenizer.encode(content)
    out, state, k_list, v_list = model.forward(tokens, None)
    # k_list, v_list: [L, T, D]
    memory_loss = calculate_memory_loss(state, k_list, v_list)
    category2loss[line['category']].append(memory_loss.item())
# print results
total_avg_loss = 0
for category, losses in category2loss.items():
    avg_loss = sum(losses) / len(losses)
    print(f"Category: {category}, Average Memory Loss: {avg_loss:.4f}")
    total_avg_loss += avg_loss
print(f"Overall Average Memory Loss: {total_avg_loss / len(category2loss):.4f}")
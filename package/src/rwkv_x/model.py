########################################################################################################
# The RWKV-X Language Model - https://github.com/add_later
########################################################################################################

from dataclasses import dataclass
import os, types
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

current_path = os.path.dirname(os.path.abspath(__file__))

MyModule = torch.nn.Module
def __nop(ob):
    return ob
MyFunction = __nop
MyStatic = __nop

DTYPE = None
DEVICE = None
HEAD_SIZE = 64


# CUDA 加速模块
if os.environ.get('RWKV_CUDA_ON') == '1':
    from torch.utils.cpp_extension import load
    load(
        name="wkv7s",
        sources=[f"{current_path}/cuda/rwkv7_op.cpp", f"{current_path}/cuda/rwkv7.cu"],
        is_python_module=False,
        verbose=True,
        extra_cuda_cflags=[
            "-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3",
            "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"
        ]
    )

    class WKV_7(torch.autograd.Function):
        @staticmethod
        def forward(ctx, state, r, w, k, v, a, b):
            T, C = r.size()
            H = C // HEAD_SIZE
            y = torch.empty((T, C), device=DEVICE, dtype=r.dtype)
            if DTYPE == torch.float16:
                torch.ops.wkv7s.forward_fp16(1, T, C, H, state, r, w, k, v, a, b, y)
            elif DTYPE == torch.bfloat16:
                torch.ops.wkv7s.forward_bf16(1, T, C, H, state, r, w, k, v, a, b, y)
            elif DTYPE == torch.float32:
                torch.ops.wkv7s.forward_fp32(1, T, C, H, state, r, w, k, v, a, b, y)
            return y

    def RWKV7_OP(state, r, w, k, v, a, b):
        return WKV_7.apply(state, r, w, k, v, a, b)


########################################################################################################

@MyStatic
def RWKV_x070_TMix_one(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    xx = x_prev - x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = torch.nn.functional.normalize((k * k_k).view(H,N), dim=-1, p=2.0).view(H*N)
    k = k * (1 + (a-1) * k_a)
    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)
    w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float())) # 0.606531 = exp(-0.5)

    vk = v.view(H,N,1) @ k.view(H,1,N)
    ab = (-kk).view(H,N,1) @ (kk*a).view(H,1,N)
    state = state * w.view(H,1,N) + state @ ab.float() + vk.float()
    xx = (state.to(dtype=x.dtype) @ r.view(H,N,1))

    xx = torch.nn.functional.group_norm(xx.view(1,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(H*N)    
    xx = xx + ((r * k * r_k).view(H,N).sum(dim=-1, keepdim=True) * v.view(H,N)).view(H*N)
    return (xx * g) @ O_, x, state, v_first

if os.environ.get('RWKV_CUDA_ON') == '1':
    @MyStatic
    def RWKV_x070_TMix_seq(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
        T = x.shape[0]
        xx = torch.cat((x_prev.unsqueeze(0), x[:-1,:])) - x
        xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

        r = xr @ R_
        w = torch.tanh(xw @ w1) @ w2
        k = xk @ K_
        v = xv @ V_
        a = torch.sigmoid(a0 + (xa @ a1) @ a2)
        g = torch.sigmoid(xg @ g1) @ g2

        kk = torch.nn.functional.normalize((k * k_k).view(T,H,N), dim=-1, p=2.0).view(T,H*N)
        k = k * (1 + (a-1) * k_a)
        if layer_id == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

        w = -torch.nn.functional.softplus(-(w0 + w)) - 0.5
        xx = RWKV7_OP(state, r, w, k, v, -kk, kk*a)

        xx = torch.nn.functional.group_norm(xx.view(T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(T,H*N)
        xx = xx + ((r * k * r_k).view(T,H,N).sum(dim=-1, keepdim=True) * v.view(T,H,N)).view(T,H*N)
        return (xx * g) @ O_, x[-1,:], state, v_first
else:
    @MyStatic
    def RWKV_x070_TMix_seq(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
        T = x.shape[0]
        xx = torch.cat((x_prev.unsqueeze(0), x[:-1,:])) - x
        xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

        r = xr @ R_
        w = torch.tanh(xw @ w1) @ w2
        k = xk @ K_
        v = xv @ V_
        a = torch.sigmoid(a0 + (xa @ a1) @ a2)
        g = torch.sigmoid(xg @ g1) @ g2

        kk = torch.nn.functional.normalize((k * k_k).view(T,H,N), dim=-1, p=2.0).view(T,H*N)
        k = k * (1 + (a-1) * k_a)
        if layer_id == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

        w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float())) # 0.606531 = exp(-0.5)
        for t in range(T):
            r_, w_, k_, v_, kk_, a_ = r[t], w[t], k[t], v[t], kk[t], a[t]
            vk = v_.view(H,N,1) @ k_.view(H,1,N)
            ab = (-kk_).view(H,N,1) @ (kk_*a_).view(H,1,N)
            state = state * w_.view(H,1,N) + state @ ab.float() + vk.float()
            xx[t] = (state.to(dtype=x.dtype) @ r_.view(H,N,1)).view(H*N)

        xx = torch.nn.functional.group_norm(xx.view(T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(T,H*N)
        xx = xx + ((r * k * r_k).view(T,H,N).sum(dim=-1, keepdim=True) * v.view(T,H,N)).view(T,H*N)
        return (xx * g) @ O_, x[-1,:], state, v_first



@MyStatic
def RWKV_x070_CMix_one(x, x_prev, x_k, K_, V_):
    xx = x_prev - x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return k @ V_, x

@MyStatic
def RWKV_x070_CMix_seq(x, x_prev, x_k, K_, V_):
    xx = torch.cat((x_prev.unsqueeze(0), x[:-1,:])) - x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return k @ V_, x[-1,:]



class RWKV_x070(MyModule):
    def __init__(self, model_state_dict):
        super().__init__()
        self.eval()
        args = types.SimpleNamespace()
        self.args = args
        
        self.z = model_state_dict
        z = self.z

        self.n_head, self.head_size = z['blocks.0.att.r_k'].shape
        args.head_size = self.head_size
        args.vocab_size, args.n_embd = z['emb.weight'].shape

        args.n_layer = 0
        keys = list(z.keys())
        for k in keys:
            layer_id = int(k.split('.')[1]) if ('blocks.' in k) else 0
            args.n_layer = max(args.n_layer, layer_id+1)
            if 'key.weight' in k or 'value.weight' in k or 'receptance.weight' in k or 'output.weight' in k or 'head.weight' in k:
                z[k] = z[k].t()
            z[k] = z[k].squeeze().to(dtype=DTYPE)
            if k.endswith('att.r_k'): 
                z[k] = z[k].flatten()

        self.n_embd = args.n_embd
        self.n_layer = args.n_layer

        z['emb.weight'] = F.layer_norm(z['emb.weight'], (args.n_embd,), weight=z['blocks.0.ln0.weight'], bias=z['blocks.0.ln0.bias'])
        z['blocks.0.att.v0'] = z['blocks.0.att.a0'] # actually ignored
        z['blocks.0.att.v1'] = z['blocks.0.att.a1'] # actually ignored
        z['blocks.0.att.v2'] = z['blocks.0.att.a2'] # actually ignored

        # core modification: construct block list
        self.blocks = nn.ModuleList([
            RWKVBlock(i, z, self.n_head, self.head_size, self.n_embd)
            for i in range(self.n_layer)
        ])

    @torch.inference_mode()
    def forward(self, idx, state, full_output=False):
        if state == None:
            state = [None for _ in range(self.args.n_layer * 3)]
            for i in range(self.args.n_layer): # state: 0=att_x_prev 1=att_kv 2=ffn_x_prev
                state[i*3+0] = torch.zeros(self.args.n_embd, dtype=DTYPE, requires_grad=False, device=DEVICE)
                state[i*3+1] = torch.zeros((self.args.n_embd // self.args.head_size, self.args.head_size, self.args.head_size), dtype=torch.float, requires_grad=False, device=DEVICE)
                state[i*3+2] = torch.zeros(self.args.n_embd, dtype=DTYPE, requires_grad=False, device=DEVICE)

        if type(idx) is list:
            if len(idx) > 1:
                return self.forward_seq(idx, state, full_output)
            else:
                return self.forward_one(idx[0], state)
        else:
            return self.forward_one(idx, state)

    @torch.inference_mode()
    def forward_one(self, idx: int, state: List[torch.Tensor]):
        z = self.z
        x = z['emb.weight'][idx]
        v_first = torch.empty_like(x)

        for block in self.blocks:
            x, state, v_first = block.forward_one(x, state, v_first)

        x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
        x = x @ z['head.weight']
        return x, state

    @torch.inference_mode()
    def forward_seq(self, idx: List[int], state: List[torch.Tensor], full_output: bool = False):
        z = self.z
        x = z['emb.weight'][idx]
        v_first = torch.empty_like(x)

        for block in self.blocks:
            x, state, v_first = block.forward_seq(x, state, v_first)

        if not full_output:
            x = x[-1]

        x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
        x = x @ z['head.weight']
        return x, state

        

class RWKVBlock(nn.Module):
    def __init__(self, layer_id, z, n_head, head_size, n_embd):
        super().__init__()
        self.layer_id = layer_id
        self.z = z
        self.n_head = n_head
        self.head_size = head_size
        self.n_embd = n_embd

    def forward_one(self, x, state, v_first):
        i = self.layer_id
        z = self.z
        bbb, att, ffn = f'blocks.{i}.', f'blocks.{i}.att.', f'blocks.{i}.ffn.'

        xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])
        xx, state[i*3+0], state[i*3+1], v_first = RWKV_x070_TMix_one(i, self.n_head, self.head_size, xx, state[i*3+0], v_first, state[i*3+1],
            z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
            z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
            z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
            z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
            z[att+'ln_x.weight'], z[att+'ln_x.bias'])
        x = x + xx

        xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])
        xx, state[i*3+2] = RWKV_x070_CMix_one(xx, state[i*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
        x = x + xx

        return x, state, v_first

    def forward_seq(self, x, state, v_first):
        i = self.layer_id
        z = self.z
        bbb, att, ffn = f'blocks.{i}.', f'blocks.{i}.att.', f'blocks.{i}.ffn.'

        xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

        xx, state[i*3+0], state[i*3+1], v_first = RWKV_x070_TMix_seq(i, self.n_head, self.head_size, xx, state[i*3+0], v_first, state[i*3+1],
            z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
            z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
            z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
            z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
            z[att+'ln_x.weight'], z[att+'ln_x.bias'])
        x = x + xx

        xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])
        xx, state[i*3+2] = RWKV_x070_CMix_seq(xx, state[i*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
        x = x + xx

        return x, state, v_first


################################## Sparse Attention ##############################################
class CausalSparseAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.output = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.attn_chunk_size = config.attn_chunk_size
        self.attn_topk = config.attn_topk
        self.short_sequence_criteria = config.short_sequence_criteria
        # kv cache management
        self.max_kv_cache_size = config.max_kv_cache_size # condition that trigger the cache management
        self.kv_cache_window_size = config.kv_cache_window_size # observation window size
        self.min_kv_cache_size = config.min_kv_cache_size # minimum kv cache size
        self.attn_mode = config.attn_mode

    def update_kv_cache(self, q, k, v):
        '''
        input: q, k, v: (B, QT, C), (B, KT, C)
        output: k_past_compress, v_past_compress: (B, min_kv_cache_size, C)
        '''
        B, QT, C = q.size()
        KT, VT = k.size(1), v.size(1)
        assert KT == VT, "key and value must have the same length"
        # check if the cache size is larger than the minimum size
        if self.min_kv_cache_size >= KT:
            return k, v
        # split k, v into past and current
        past_KT = KT - self.kv_cache_window_size
        k_past, k_cur = k[:, :past_KT, :], k[:, past_KT: :]
        v_past, v_cur = v[:, :past_KT, :], v[:, past_KT:, :]
        q = q.view(B, QT, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, QT, hs)
        k_past = k_past.view(B, past_KT, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, past_KT, hs)
        v_past = v_past.view(B, past_KT, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, past_KT, hs)
        # calculate attention weights
        B, NH, QT, HS = q.size()
        attn_weights = torch.matmul(q, k_past.transpose(2, 3)) / math.sqrt(HS) # (B, NH, QT, past_KT)
        attn_weights = F.softmax(attn_weights, dim=-1) # (B, NH, QT, past_KT)
        vote = attn_weights.sum(dim=-2) # (B, NH, past_KT) Sum the weight along the query dimension
        assert self.min_kv_cache_size - self.kv_cache_window_size > 0
        keep_KT = min(self.min_kv_cache_size - self.kv_cache_window_size, past_KT)
        indices = vote.topk(keep_KT, dim=-1).indices
        # Expand the indices to match the head dimension for gathering
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, HS)
        k_past_compress = torch.gather(k_past, 2, indices)
        v_past_compress = torch.gather(v_past, 2, indices)
        # Reshape back to (B, KT, C)
        k_past_compress = k_past_compress.transpose(1, 2).contiguous().view(B, -1, C) # (B, keep_KT, C)
        v_past_compress = v_past_compress.transpose(1, 2).contiguous().view(B, -1, C) # (B, keep_KT, C)
        # 
        k_cache = torch.cat([k_past_compress, k_cur], dim=1) # (B, keep_KT, C)
        v_cache = torch.cat([v_past_compress, v_cur], dim=1)
        return k_cache, v_cache

    def forward_one(self, x, k_cache, v_cache):
        ''' used for decode, only one token at a time
        input: x: (C) and k, v cache (1, CT, C)
        output: y: (C) and k, v cache (1, CT+1, C)
        '''
        if len(x.shape) == 1:
            x = x.unsqueeze(0).unsqueeze(0) # (C) -> (1, 1, C)
        assert x.size(0) == 1, "batch size must be 1"
        assert x.size(1) == 1, "sequence length must be 1"
        C = x.size(-1)
        q, k, v = self.receptance(x), self.key(x), self.value(x)
        # manage k, v cache, only when not use full attention
        if self.deocoding_attn_mode != 'full' and k_cache.size(1) > self.max_kv_cache_size:
            k_cache, v_cache = self.update_kv_cache(q, k_cache, v_cache)
        CT = k_cache.size(1)
        # apply the attention, only decoding consider the full attention mode
        if CT <= self.short_sequence_criteria or self.deocoding_attn_mode == 'full': # for short sequence, use full attention
            k_cache = torch.cat((k_cache, k), dim=1) # update k cache
            v_cache = torch.cat((v_cache, v), dim=1) # update v cache
            q = q.view(1, 1, self.n_head, C // self.n_head).transpose(1, 2) # (1, 1, C) -> (1, nh, 1, hs)
            k_comb = k_cache.view(1, CT+1, self.n_head, C // self.n_head).transpose(1, 2) # (1, nh, CT+1, hs)
            v_comb = v_cache.view(1, CT+1, self.n_head, C // self.n_head).transpose(1, 2) # (1, nh, CT+1, hs)
            # !!! super be careful here, the attention is not causal !!!
            # is_causal=True -> torch.ones(L, S, dtype=torch.bool).tril(diagonal=0) -> [1， S] -> [1, 0, 0, 0, ...]
            y = F.scaled_dot_product_attention(q, k_comb, v_comb)
            y = y.transpose(1, 2).contiguous().view(C) # (1, nh, 1, hs) -> (1, 1, nh, hs) -> (C)
            y = self.output(y)
            return y, k_cache, v_cache
        else:
            reminder = CT % self.attn_chunk_size
            k_chunk, k_reminder = k_cache[:, :CT-reminder, :], k_cache[:, CT-reminder:, :]
            v_chunk, v_reminder = v_cache[:, :CT-reminder, :], v_cache[:, CT-reminder:, :]
            # split k, v into chunks
            num_chunks = k_chunk.size(1) // self.attn_chunk_size
            k_chunk = k_chunk.view(1, num_chunks, self.attn_chunk_size, C)
            v_chunk = v_chunk.view(1, num_chunks, self.attn_chunk_size, C)
            # get chunk key
            chunk_key = k_chunk.mean(dim=2) # (1, num_chunks, C)
            # get top-k chunk
            chunk_score = torch.einsum('bqc,bkc->bqk', q, chunk_key) # (1, 1, num_chunks)
            topk_chunk_indices = torch.topk(chunk_score, self.attn_topk, dim=-1).indices.squeeze() # (attn_topk)
            # get top-k chunk key and value
            topk_k = k_chunk[:, topk_chunk_indices].view(1, -1, C) # (1, attn_topk * attn_chunk_size, C)
            topk_v = v_chunk[:, topk_chunk_indices].view(1, -1, C) # (1, attn_topk * attn_chunk_size, C)
            # combine with reminder and current k, v
            k_comb = torch.cat((topk_k, k_reminder, k), dim=1)
            v_comb = torch.cat((topk_v, v_reminder, v), dim=1)
            # calculate attention and output
            new_T = k_comb.size(1)
            q = q.view(1, 1, self.n_head, C // self.n_head).transpose(1, 2)
            k_comb = k_comb.view(1, new_T, self.n_head, C // self.n_head).transpose(1, 2)
            v_comb = v_comb.view(1, new_T, self.n_head, C // self.n_head).transpose(1, 2)
            # !!! super be careful, the attention here can not be causal !!!
            y = F.scaled_dot_product_attention(q, k_comb, v_comb)
            y = y.transpose(1, 2).contiguous().view(C) # (1, 1, C) -> (C)
            y = self.output(y)
            k_cache = torch.cat((k_cache, k), dim=1)
            v_cache = torch.cat((v_cache, v), dim=1)
            return y, k_cache, v_cache

    def forward_seq(self, x, k_cache, v_cache):
        '''
        input: x: (T, C) and k, v cache (1, CT, C)
        output: y: (T, C) and k, v cache (1, CT+T, C)
        '''
        if len(x.shape) == 2:
            x = x.unsqueeze(0) # (T, C) -> (1, T, C)
        B, T, C = x.size()
        # manage k, v cache, only when not use full attention
        if self.attn_mode != 'full' and k_cache.size(1) > self.max_kv_cache_size:
            k_cache, v_cache = self.update_kv_cache(x, k_cache, v_cache)
        CT = k_cache.size(1) # cache seq length
        # apply the attention
        if (T+CT) <= self.short_sequence_criteria or self.prefill_attn_mode == 'full': # for short sequence, use full attention
            q, k, v = self.receptance(x), self.key(x), self.value(x)
            # prefix mask
            causal_mask = torch.ones((T, T), dtype=torch.bool, device=x.device).tril(diagonal=0)
            cache_mask = torch.ones((T, CT), dtype=torch.bool, device=x.device)
            prefix_mask = torch.cat((cache_mask, causal_mask), dim=1)
            k_cache = torch.cat((k_cache, k), dim=1) # update k cache
            v_cache = torch.cat((v_cache, v), dim=1) # update v cache
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            k_comb = k_cache.view(B, CT+T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, CT+T, hs)
            v_comb = v_cache.view(B, CT+T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, CT+T, hs)
            y = F.scaled_dot_product_attention(q, k_comb, v_comb, attn_mask=prefix_mask)
            y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
            # output projection
            y = self.output(y).squeeze(0) # (1, T, C) -> (T, C)
            return y, k_cache, v_cache
        else: # chunked attention ignoring the kv cache
            # pad on right side to match the chunk size
            pad_size = self.attn_chunk_size - (T % self.attn_chunk_size)
            if pad_size != self.attn_chunk_size:
                zero_pad = torch.zeros((B, pad_size, C), dtype=x.dtype, device=x.device)
                x = torch.cat((x, zero_pad), dim=1)
                T = x.size(1) # update T after padding
            # fold x into chunks, merge with batch size dim
            num_chunks = T // self.attn_chunk_size
            x = x.view(B * num_chunks, self.attn_chunk_size, C)
            new_B, new_T, C = x.size()
            q, k, v = self.receptance(x), self.key(x), self.value(x)
            k = k.view(new_B, new_T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)        
            q = q.view(new_B, new_T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(new_B, new_T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # here we use causal attention in chunk
            y = y.transpose(1, 2).contiguous().view(new_B, new_T, C) # re-assemble all head outputs side by side
            k = k.transpose(1, 2).contiguous().view(new_B, new_T, C)
            v = v.transpose(1, 2).contiguous().view(new_B, new_T, C)
            # output projection
            y = self.output(y)
            # unfold y, k, v into original shape
            y = y.view(B, num_chunks*self.attn_chunk_size, C)
            k = k.view(B, num_chunks*self.attn_chunk_size, C)
            v = v.view(B, num_chunks*self.attn_chunk_size, C)
            # remove padding
            if pad_size != self.attn_chunk_size:
                y = y[:, :-pad_size, :]
                k = k[:, :-pad_size, :]
                v = v[:, :-pad_size, :]
            # remove extra dimension
            y = y.squeeze(0)
            # update k, v cache
            k_cache = torch.cat((k_cache, k), dim=1)
            v_cache = torch.cat((v_cache, v), dim=1)
            return y, k_cache, v_cache


########################################################################################################
# RWKV ChannelMix
########################################################################################################
class RWKV_CMix_x070(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Linear(n_embd, n_embd * 4, bias=False)
        self.value = nn.Linear(n_embd * 4, n_embd, bias=False)
        self.x_k = nn.Parameter(torch.ones(1, 1, n_embd))
    
    def forward(self, x, x_prev):
        if len(x.shape) == 1:
            xx = x_prev - x
            k = x + xx * self.x_k.squeeze()
            k = torch.relu(self.key(k)) ** 2
            return self.value(k), x
        
        xx = torch.cat((x_prev.unsqueeze(0), x[:-1,:])) - x
        k = x + xx * self.x_k.squeeze(0) # (1, 1, C) -> (1, C)
        k = torch.relu(self.key(k)) ** 2
        return self.value(k), x[-1,:]
    

class SparseAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.att = CausalSparseAttention(config)
        self.ffn = RWKV_CMix_x070(config.n_embd)
        
    def forward_seq(self, x, state):
        xx, state['k_cache'], state['v_cache'] = self.att.forward_seq(
            self.ln1(x), state['k_cache'], state['v_cache']
            )
        x = x + xx
        xx, state['ffn_x_prev'] = self.ffn(self.ln2(x), state['ffn_x_prev'])
        x = x + xx
        return x, state
    
    def forward_one(self, x, state):
        '''
        x: (C)
        '''
        xx, state['k_cache'], state['v_cache'] = self.att.forward_one(
            self.ln1(x), state['k_cache'], state['v_cache']
            )
        x = x + xx
        xx, state['ffn_x_prev'] = self.ffn(self.ln2(x), state['ffn_x_prev'])
        x = x + xx
        return x, state

@dataclass
class RWKV_X_Config:
    n_rwkv_layer: int = 0
    n_attn_layer: int = 0
    n_head: int = 0
    n_embd: int = 0
    head_size: int = 64
    attn_chunk_size: int = 2000
    attn_topk: int = 3
    short_sequence_criteria: int = 8000 # if sequence length <= this, use full attention
    max_kv_cache_size: int = 20000
    kv_cache_window_size: int = 2000
    min_kv_cache_size: int = 16000
    prefill_attn_mode: str = 'chunk' # 'chunk' or 'full'
    deocoding_attn_mode: str = 'full' # 'full' or 'snapKV' or 'new'

class RWKV_X(nn.Module):
    def __init__(self, model_path, strategy, config=None):
        super().__init__()
        print(f'Loading {model_path} ({strategy})\n')
        rwkv_state_dict, attn_state_dict, config = self.load_from_ckpt(model_path, strategy, config)
        self.rwkv = RWKV_x070(rwkv_state_dict).to(device=DEVICE).to(DTYPE)
        self.attn = nn.ModuleList([SparseAttentionBlock(config) for _ in range(config.n_attn_layer)])
        self.attn.to(device=DEVICE).to(DTYPE)
        self.attn.load_state_dict(attn_state_dict, strict=True)
        self.config = config

    def load_from_ckpt(self, model_path, strategy, config=None):
        global DTYPE, DEVICE
        ss = strategy.split(' ')
        DEVICE = ss[0]
        if ss[1] == 'fp16':
            DTYPE = torch.half
        elif ss[1] == 'fp32':
            DTYPE = torch.float32
        elif ss[1] == 'bf16':
            DTYPE = torch.bfloat16
        else:
            assert False, "currently rwkv-x strategy must be: cuda/cpu fp16/fp32/bf16"

        # Load the model from the checkpoint
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=DEVICE, weights_only=True)
        else:
            raise FileNotFoundError(f"Model path {model_path} does not exist.")

        rwkv_state_dict = {k[5:]: v for k, v in ckpt.items() if k.startswith("rwkv.")}
        attn_state_dict = {k[5:]: v for k, v in ckpt.items() if k.startswith("moba.")}

        n_embd = rwkv_state_dict['emb.weight'].shape[1]
        n_head = n_embd // 64
        n_rwkv_layer = len({k.split('.')[1] for k in rwkv_state_dict if k.startswith("blocks.")})
        n_attn_layer = len({k.split('.')[0] for k in attn_state_dict if k[0].isdigit()})
        if config is None:
            config = RWKV_X_Config(
                n_rwkv_layer=n_rwkv_layer,
                n_attn_layer=n_attn_layer,
                n_head=n_head,
                n_embd=n_embd,
            )
        else:
            config = RWKV_X_Config(
                n_rwkv_layer=n_rwkv_layer,
                n_attn_layer=n_attn_layer,
                n_head=n_head,
                n_embd=n_embd,
                attn_chunk_size=config.attn_chunk_size,
                attn_topk=config.attn_topk,
                max_kv_cache_size=config.max_kv_cache_size,
                prefill_attn_mode=config.prefill_attn_mode,
                deocoding_attn_mode=config.deocoding_attn_mode,
                short_sequence_criteria=config.short_sequence_criteria,
                kv_cache_window_size=config.kv_cache_window_size,
                min_kv_cache_size=config.min_kv_cache_size,
                head_size=config.head_size
            )
        return rwkv_state_dict, attn_state_dict, config


    @torch.inference_mode()
    def forward(self, idx, state, full_output=False):
        if state == None:
            rwkv_state = [None for _ in range(self.config.n_rwkv_layer * 3)]
            for i in range(self.config.n_rwkv_layer): # state: 0=att_x_prev 1=att_kv 2=ffn_x_prev
                rwkv_state[i*3+0] = torch.zeros(self.config.n_embd, dtype=DTYPE, requires_grad=False, device=DEVICE)
                rwkv_state[i*3+1] = torch.zeros((self.config.n_embd // self.config.head_size, self.config.head_size, self.config.head_size), dtype=torch.float, requires_grad=False, device=DEVICE)
                rwkv_state[i*3+2] = torch.zeros(self.config.n_embd, dtype=DTYPE, requires_grad=False, device=DEVICE)
            attn_state = []
            for _ in range(self.config.n_attn_layer):
                attn_state.append({
                    'k_cache': torch.zeros((1, 0, self.config.n_embd), dtype=DTYPE, device=DEVICE, requires_grad=False),
                    'v_cache': torch.zeros((1, 0, self.config.n_embd), dtype=DTYPE, device=DEVICE, requires_grad=False),
                    'ffn_x_prev':  torch.zeros(self.config.n_embd, dtype=DTYPE, requires_grad=False, device=DEVICE)
                })
            state = {'rwkv': rwkv_state, 'attn': attn_state}

        if type(idx) is list:
            if len(idx) > 1:
                return self.forward_seq(idx, state, full_output)
            else:
                return self.forward_one(idx[0], state)
        else:
            return self.forward_one(idx, state)
        
    def forward_one(self, idx: int, state: dict):
        z = self.rwkv.z
        x = z['emb.weight'][idx]
        v_first = torch.empty_like(x)

        rwkv_id, attn_id = 0, 0
        for block in self.get_block_exe_order():
            if isinstance(block, RWKVBlock):
                x, state['rwkv'], v_first = block.forward_one(x, state['rwkv'], v_first)
                rwkv_id += 1
            else:
                x, state['attn'][attn_id] = block.forward_one(x, state['attn'][attn_id])
                attn_id += 1

        x = F.layer_norm(x, (self.config.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
        x = x @ z['head.weight']
        return x, state
    
    def forward_seq(self, idx: List[int], state: dict, full_output=False):
        z = self.rwkv.z
        x = z['emb.weight'][idx]
        v_first = torch.empty_like(x)

        rwkv_id, attn_id = 0, 0
        for block in self.get_block_exe_order():
            if isinstance(block, RWKVBlock):
                x, state['rwkv'], v_first = block.forward_seq(x, state['rwkv'], v_first)
                rwkv_id += 1
            else:
                x, state['attn'][attn_id] = block.forward_seq(x, state['attn'][attn_id])
                attn_id += 1

        if not full_output:
            x = x[-1]
        x = F.layer_norm(x, (self.config.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
        x = x @ z['head.weight']
        return x, state
    
    def get_block_exe_order(self):
        if self.config.n_attn_layer == 0:
            return self.rwkv.blocks
        if self.config.n_attn_layer == 1:
            blocks = self.rwkv.blocks + self.attn
            return blocks
        interval = len(self.rwkv.blocks) // self.config.n_attn_layer # 12 // 4 = 3
        blocks = [] # [RWKVBlock * interval, AttnBlock, RWKVBlock * interval, AttnBlock, ...]
        for i in range(self.config.n_attn_layer):
            blocks += self.rwkv.blocks[i * interval: (i + 1) * interval]
            blocks.append(self.attn[i])
        return blocks

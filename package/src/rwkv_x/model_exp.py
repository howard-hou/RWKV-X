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


# CUDA
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
    return (xx * g) @ O_, x, state, v_first, k, v

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
        return (xx * g) @ O_, x[-1,:], state, v_first, k, v
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
        return (xx * g) @ O_, x[-1,:], state, v_first, k, v



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
    def __init__(self, model, strategy):
        global DTYPE, DEVICE
        super().__init__()
        self.eval()
        args = types.SimpleNamespace()
        self.args = args
        args.MODEL_NAME = model

        print(f'Loading {model} ({strategy})\n')

        ss = strategy.split(' ')
        DEVICE = ss[0]
        if ss[1] == 'fp16':
            DTYPE = torch.half
        elif ss[1] == 'fp32':
            DTYPE = torch.float32
        elif ss[1] == 'bf16':
            DTYPE = torch.bfloat16
        else:
            assert False, "currently rwkv7 strategy must be: cuda/cpu fp16/fp32/bf16"
        
        temp_z = torch.load(args.MODEL_NAME + '.pth', map_location='cpu', mmap=True)

        self.n_head, self.head_size = temp_z['blocks.0.att.r_k'].shape
        args.head_size = self.head_size
        args.vocab_size, args.n_embd = temp_z['emb.weight'].shape

        args.n_layer = 0
        keys = list(temp_z.keys())
        self.z = {}

        for k in keys:
            layer_id = int(k.split('.')[1]) if ('blocks.' in k) else 0
            args.n_layer = max(args.n_layer, layer_id+1)
            tensor = temp_z[k]
            if 'key.weight' in k or 'value.weight' in k or 'receptance.weight' in k or 'output.weight' in k or 'head.weight' in k:
                tensor = tensor.t().contiguous()
            tensor = tensor.squeeze()
            if k.endswith('att.r_k'): 
                tensor = tensor.flatten()
            self.z[k] = tensor.to(DEVICE).to(DTYPE)
            del temp_z[k]
            if keys.index(k) % 5 == 0:
                torch.cuda.empty_cache()

        self.n_embd = args.n_embd
        self.n_layer = args.n_layer

        self.z['emb.weight'] = F.layer_norm(self.z['emb.weight'], (args.n_embd,), weight=self.z['blocks.0.ln0.weight'], bias=self.z['blocks.0.ln0.bias'])
        self.z['blocks.0.att.v0'] = torch.empty(0, device=DEVICE, dtype=DTYPE) # actually ignored
        self.z['blocks.0.att.v1'] = torch.empty(0, device=DEVICE, dtype=DTYPE) # actually ignored
        self.z['blocks.0.att.v2'] = torch.empty(0, device=DEVICE, dtype=DTYPE) # actually ignored

        # core modification: construct block list
        self.blocks = nn.ModuleList([
            RWKVBlock(i, self.z, self.n_head, self.head_size, self.n_embd)
            for i in range(self.n_layer)
        ])
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def forward_one(self, idx: int, state: List[torch.Tensor]):
        z = self.z
        x = z['emb.weight'][idx]
        v_first = torch.empty_like(x)

        for block in self.blocks:
            x, state, v_first, k, v = block.forward_one(x, state, v_first)

        x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
        x = x @ z['head.weight']
        return x, state, k, v

    @torch.inference_mode()
    def forward_seq(self, idx: List[int], state: List[torch.Tensor], full_output: bool = False):
        z = self.z
        x = z['emb.weight'][idx]
        v_first = torch.empty_like(x)

        for block in self.blocks:
            x, state, v_first, k, v = block.forward_seq(x, state, v_first)

        if not full_output:
            x = x[-1]

        x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
        x = x @ z['head.weight']
        return x, state, k, v


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
        xx, state[i*3+0], state[i*3+1], v_first, k, v = RWKV_x070_TMix_one(i, self.n_head, self.head_size, xx, state[i*3+0], v_first, state[i*3+1],
            z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
            z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
            z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
            z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
            z[att+'ln_x.weight'], z[att+'ln_x.bias'])
        x = x + xx

        xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])
        xx, state[i*3+2] = RWKV_x070_CMix_one(xx, state[i*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
        x = x + xx

        return x, state, v_first, k, v

    def forward_seq(self, x, state, v_first):
        i = self.layer_id
        z = self.z
        bbb, att, ffn = f'blocks.{i}.', f'blocks.{i}.att.', f'blocks.{i}.ffn.'

        xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

        xx, state[i*3+0], state[i*3+1], v_first, k, v = RWKV_x070_TMix_seq(i, self.n_head, self.head_size, xx, state[i*3+0], v_first, state[i*3+1],
            z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
            z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
            z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
            z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
            z[att+'ln_x.weight'], z[att+'ln_x.bias'])
        x = x + xx

        xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])
        xx, state[i*3+2] = RWKV_x070_CMix_seq(xx, state[i*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
        x = x + xx

        return x, state, v_first, k, v


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
    

class RWKV_EXP(nn.Module):
    def __init__(self, model_path, strategy):
        super().__init__()
        model_path = model_path.replace('.pth', '')
        self.rwkv = RWKV_x070(model_path, strategy)
        self.args = self.rwkv.args

    def generate_zero_state(self):
        state = [None for _ in range(self.args.n_layer * 3)]
        for i in range(self.args.n_layer): # state: 0=att_x_prev 1=att_kv 2=ffn_x_prev
            state[i*3+0] = torch.zeros(self.args.n_embd, dtype=DTYPE, requires_grad=False, device=DEVICE)
            state[i*3+1] = torch.zeros((self.args.n_embd // self.args.head_size, self.args.head_size, self.args.head_size), dtype=torch.float, requires_grad=False, device=DEVICE)
            state[i*3+2] = torch.zeros(self.args.n_embd, dtype=DTYPE, requires_grad=False, device=DEVICE)
        return state

    @torch.inference_mode()
    def forward(self, idx, state, full_output=False):
        if state == None:
            state = self.generate_zero_state()

        if type(idx) is list:
            if len(idx) > 1:
                return self.forward_seq(idx, state, full_output)
            else:
                return self.forward_one(idx[0], state)
        else:
            return self.forward_one(idx, state)
        
    def forward_one(self, idx: int, state: list):
        z = self.rwkv.z
        x = z['emb.weight'][idx]
        v_first = torch.empty_like(x)

        k_list, v_list = [], []
        for block in self.rwkv.blocks:
            x, state, v_first, k, v = block.forward_one(x, state, v_first)
            k_list.append(k)
            v_list.append(v)

        x = F.layer_norm(x, (self.args.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
        x = x @ z['head.weight']
        return x, state, k_list, v_list
    
    def forward_seq(self, idx: List[int], state: list, full_output=False):
        z = self.rwkv.z
        x = z['emb.weight'][idx]
        v_first = torch.empty_like(x)

        k_list, v_list = [], []
        for block in self.rwkv.blocks:
            x, state, v_first, k, v = block.forward_seq(x, state, v_first)
            k_list.append(k)
            v_list.append(v)
            
        if not full_output:
            x = x[-1]
        x = F.layer_norm(x, (self.args.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
        x = x @ z['head.weight']
        return x, state, k_list, v_list

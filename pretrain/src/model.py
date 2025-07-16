########################################################################################################
# The RWKV-X Language Model - https://github.com/howard-hou/RWKV-X
########################################################################################################

import os, math, gc, importlib
import torch
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

# from deepspeed.runtime.fp16.onebit.zoadam import ZeroOneAdam
from .utils import compress_parameter_names
from .moba_efficient import moba_attn_varlen

def __nop(ob):
    return ob

STOP_TOKEN_INDEX = 261

MyModule = nn.Module
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method

########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])
CHUNK_LEN = 16
flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
load(name="wind_backstepping", sources=[f'cuda/wkv7_cuda.cu', 'cuda/wkv7_op.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)

class WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w,q,k,v,z,b):
        B,T,H,C = w.shape 
        assert T%CHUNK_LEN == 0
        assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
        assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
        y = torch.empty_like(v)
        s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
        sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
        torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
        ctx.save_for_backward(w,q,k,v,z,b,s,sa)
        return y
    @staticmethod
    def backward(ctx, dy):
        assert all(i.dtype==torch.bfloat16 for i in [dy])
        assert all(i.is_contiguous() for i in [dy])
        w,q,k,v,z,b,s,sa = ctx.saved_tensors
        dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
        torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
        return dw,dq,dk,dv,dz,db

def RUN_CUDA_RWKV7g(q,w,k,v,a,b):
    B,T,HC = q.shape
    q,w,k,v,a,b = [i.view(B,T,HC//64,64) for i in [q,w,k,v,a,b]]
    return WindBackstepping.apply(w,q,k,v,a,b).view(B,T,HC)

########################################################################################################
# RWKV TimeMix
########################################################################################################

class RWKV_Tmix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.x_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            # D_DECAY_LORA = 64
            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

            # D_AAA_LORA = 64
            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C))

            # D_MV_LORA = 32
            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            if self.layer_id != 0: # not needed for the first layer
                self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
                self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
                self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)

            # D_GATE_LORA = 128
            D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion
            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
            self.k_a = nn.Parameter(torch.ones(1,1,C))
            self.r_k = nn.Parameter(torch.zeros(H,N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=(1e-5)*(args.head_size_divisor**2)) # !!! notice eps value !!!

            # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
            self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.output.weight.data.zero_()


    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first

########################################################################################################
# RWKV ChannelMix
########################################################################################################
class RWKV_CMix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

        # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
        self.key.weight.data.uniform_(-0.5/(args.n_embd**0.5), 0.5/(args.n_embd**0.5))
        self.value.weight.data.zero_()

    def forward(self, x):
        xx = self.time_shift(x) - x
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)
    
########################################################################################################
# RWKV Block
########################################################################################################

class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd) # only used in block 0, should be fused with emb
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)
        
    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)

        xx, v_first = self.att(self.ln1(x), v_first)
        x = x + xx
        x = x + self.ffn(self.ln2(x))
        return x, v_first

########################################################################################################
# MOBA Block
########################################################################################################
def hf_to_fa(x: torch.Tensor):
    """
    Args:
        x (torch.Tensor): [batch, heads, seqlen, head_dim]

    Returns:
        torch.Tensor: [batch * seqlen, heads, head_dim]
    """
    return x.permute(0, 2, 1, 3).reshape(-1, x.shape[1], x.shape[3])

class CausalSelfAttention(nn.Module):

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
        self.moba_chunk_size = config.moba_chunk_size
        self.moba_topk = config.moba_topk
        self.window_size = config.moba_chunk_size * config.moba_topk
        # zero init c_proj
        self.receptance.weight.data.uniform_(-0.5/(config.n_embd**0.5), 0.5/(config.n_embd**0.5))
        self.key.weight.data.uniform_(-0.05/(config.n_embd**0.5), 0.05/(config.n_embd**0.5))
        self.value.weight.data.uniform_(-0.5/(config.n_embd**0.5), 0.5/(config.n_embd**0.5))
        self.output.weight.data.zero_()

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        return self.short_forward(x) if T <= self.window_size else self.long_forward(x)

    def short_forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        q, k, v = self.receptance(x), self.key(x), self.value(x)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.output(y)
        return y

    def long_forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        q, k, v = self.receptance(x), self.key(x), self.value(x)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # to flash attention format
        q = hf_to_fa(q)
        k = hf_to_fa(k)
        v = hf_to_fa(v)
        cu_seqlens_k = torch.cumsum(
            torch.tensor([0] + [T] * B, device=x.device),
            dim=0,
            dtype=torch.int32,
        )
        # apply moba
        y = moba_attn_varlen(
            q=q,
            k=k,
            v=v,
            cu_seqlens=cu_seqlens_k,
            max_seqlen=T,
            moba_chunk_size=self.moba_chunk_size,
            moba_topk=self.moba_topk,
        )
        y = y.view(B, -1, x.shape[1], x.shape[2]).permute(0, 2, 1, 3) #  [batch, heads, seqlen, head_dim]
        y = y.transpose(1, 2).contiguous().view(B, T, C) # [B, T, C]
        # output projection
        y = self.output(y)
        return y

class MOBABlock(nn.Module):
    def __init__(self, args, layer_id, config):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        self.att = CausalSelfAttention(config)
        self.ffn = RWKV_CMix_x070(args, layer_id)
        
    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


class RWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)


class RWKVHybrid(pl.LightningModule):
    def __init__(self, rwkv, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.rwkv = rwkv
        self.moba = nn.ModuleList([MOBABlock(args, i, config) for i in range(config.n_moba_layer)])
        # initialize moba from rwkv
        self.init_moba_from_rwkv()
        self.use_longce = args.use_longce
        if self.use_longce:
            rank_zero_info(f"Using LongCE Loss for training.")

    def init_moba_from_rwkv(self):
        ratio = len(self.rwkv.blocks) // self.config.n_moba_layer
        # copy weights from rwkv to moba
        for i in range(self.config.n_moba_layer):
            moba_block = self.moba[i]
            rwkv_block = self.rwkv.blocks[(i + 1) * ratio - 1]
            moba_block.att.receptance.weight.data.copy_(rwkv_block.att.receptance.weight.data)
            moba_block.att.key.weight.data.copy_(rwkv_block.att.key.weight.data)
            moba_block.att.value.weight.data.copy_(rwkv_block.att.value.weight.data)
            moba_block.att.output.weight.data.copy_(rwkv_block.att.output.weight.data)
            moba_block.ffn.key.weight.data.copy_(rwkv_block.ffn.key.weight.data)
            moba_block.ffn.value.weight.data.copy_(rwkv_block.ffn.value.weight.data)

    def configure_optimizers(self):
        zero_weight_decay_group = [p for p in self.parameters() if len(p.squeeze().shape) < 2 and p.requires_grad]
        # add weight decay to len(p.squeeze().shape) >= 2
        weight_decay_group = [p for p in self.parameters() if len(p.squeeze().shape) >= 2 and p.requires_grad] 

        name_of_trainable_params = [n for n, p in self.named_parameters() if p.requires_grad]
        compressed_name_of_trainable_params = compress_parameter_names(name_of_trainable_params)
        rank_zero_info(f"Name of trainable parameters in optimizers: {compressed_name_of_trainable_params}")
        rank_zero_info(f"Number of trainable parameters in optimizers: {len(name_of_trainable_params)}")
        optim_groups = []
        optim_groups = []
        if zero_weight_decay_group:
            optim_groups += [{"params": zero_weight_decay_group, "weight_decay": 0.0}]
        if weight_decay_group:
            if self.args.weight_decay > 0:
                optim_groups += [{"params": weight_decay_group, "weight_decay": self.args.weight_decay}]
                rank_zero_info(f"Number of parameters with weight decay: {len(weight_decay_group)}, with value: {self.args.weight_decay}")
            else:
                optim_groups += [{"params": weight_decay_group, "weight_decay": 0.0}]
        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
        return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def get_block_exe_order(self):
        if self.config.n_moba_layer == 0:
            return self.rwkv.blocks
        if self.config.n_moba_layer == 1:
            blocks = self.rwkv.blocks + self.moba
            return blocks
        interval = len(self.rwkv.blocks) // self.config.n_moba_layer # 12 // 4 = 3
        blocks = [] # [RWKVBlock * interval, MOBABlock, RWKVBlock * interval, MOBABlock, ...]
        for i in range(self.config.n_moba_layer):
            blocks += self.rwkv.blocks[i * interval: (i + 1) * interval]
            blocks.append(self.moba[i])
        return blocks

    def forward(self, x):
        args = self.args
        num_tokens_to_pad = (
            CHUNK_LEN - x.size(1) % CHUNK_LEN if x.size(1) % CHUNK_LEN != 0 else 0
        )
        x = self.pad_left(x, num_tokens_to_pad)
        x = self.rwkv.emb(x)

        if args.dropout > 0:
            x = self.rwkv.drop0(x)

        v_first = torch.empty_like(x)
        block_exe_order = self.get_block_exe_order()
        for block in block_exe_order:
            if isinstance(block, MOBABlock):
                if args.grad_cp == 1:
                    x = deepspeed.checkpointing.checkpoint(block, x)
                else:
                    x = block(x)
            else:
                if args.grad_cp == 1:
                    x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first)
                else:
                    x, v_first = block(x, v_first)

        x = self.rwkv.ln_out(x)
        x = self.rwkv.head(x)
        return self.unpad(x, num_tokens_to_pad)
    
    def pad_left(self, x, num_tokens_to_pad):
        # pad left with eos token idx
        if num_tokens_to_pad != 0:
            # left padding by add eos token at the beginning
            eos_idx = torch.full(
                (x.size(0), num_tokens_to_pad),
                STOP_TOKEN_INDEX,
                dtype=torch.long,
                device=x.device,
            )
            x = torch.cat((eos_idx, x), dim=1)
        return x

    def unpad(self, x, num_tokens_to_pad):
        # unpad
        if num_tokens_to_pad > 0:
            x = x[:, num_tokens_to_pad:]
        return x
    
    def compute_longce_loss(self, input_ids, targets, trunc_len=4000, internal=4000, thre=5):
        """
        LongCE: Dynamically adjusts token-wise loss weights based on the discrepancy 
        between global and truncated local sequence losses.
        Assumes input_ids and targets are already positionally aligned.
        """
        batch_size, seq_len = input_ids.shape

        # Step 1: Compute full-sequence logits and token-level loss
        logits = self(input_ids)  # [B, L, V]
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        loss_full = loss_fn(logits.view(-1, logits.size(-1)),
                            targets.view(-1)).view(batch_size, seq_len)

        # Initialize all weights to 1
        weight = torch.ones_like(loss_full)

        with torch.no_grad():
            for start in range(trunc_len, seq_len - trunc_len, internal):
                end = min(start + trunc_len + internal, seq_len)
                int_eff = end - (start + trunc_len)
                if int_eff <= 0:
                    continue

                # Extract truncated input chunk: [B, S]
                input_chunk = input_ids[:, start:end]
                targets_chunk = targets[:, start:end]

                logits_chunk = self(input_chunk)  # [B, S, V]
                loss_chunk = loss_fn(logits_chunk[:, trunc_len: trunc_len + int_eff, :].reshape(-1, logits_chunk.size(-1)),
                                     targets_chunk[:, trunc_len: trunc_len + int_eff].reshape(-1)).view(batch_size, int_eff)

                # Get corresponding slice from full loss
                full_loss_slice = loss_full[:, start + trunc_len: start + trunc_len + int_eff]

                # Compute exponential discrepancy and clamp it
                discrepancy = torch.exp(loss_chunk - full_loss_slice)
                discrepancy = torch.clamp(discrepancy, max=thre)

                # Assign the discrepancy as weight
                weight[:, start + trunc_len: start + trunc_len + int_eff] = discrepancy

        # Apply weights to full loss and average
        weighted_loss = (loss_full * weight).mean()
        return logits, weighted_loss   

    def training_step(self, batch, batch_idx):
        input_ids, targets = batch
        if self.use_longce:
            logits, loss = self.compute_longce_loss(input_ids, targets)
        else:
            logits = self(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return L2Wrap.apply(loss, logits)

    def training_step_end(self, batch_parts):
        if pl.__version__[0]!='2':
            all = self.all_gather(batch_parts)
            if self.trainer.is_global_zero:
                self.trainer.my_loss_all = all
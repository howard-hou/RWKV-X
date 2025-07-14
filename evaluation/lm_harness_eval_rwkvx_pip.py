########################################################################################################
# The RWKV-X Language Model - https://github.com/howard-hou/RWKV-X
########################################################################################################
#
# pip install rwkv lm_eval --upgrade
# previous version only support lm_eval==0.3.0
# this version support lm_eval>=0.4.0
#
import os, math
from tqdm import tqdm
from dataclasses import dataclass
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
from torch.nn import functional as F

os.environ["RWKV_JIT_ON"] = '0'
os.environ["RWKV_CUDA_ON"] = '1'
os.environ["RWKV_V7_ON"] = "1"
from rwkv_x.model import RWKV_X
from rwkv_x.utils import PIPELINE

from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate
from lm_eval.models.huggingface import HFLM

########################################################################################################

logitBuf = {}
correctBuf = {}

@dataclass
class TokenizerOutput:
    input_ids: torch.Tensor

class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token_id = 0

    def encode(self, string: str, add_special_tokens=False):
        return self.tokenizer.encode(string)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def __call__(self, string: str):
        input_ids = torch.LongTensor(self.encode(string))
        return TokenizerOutput(input_ids=input_ids)

@register_model("RWKV-X")
class RWKVXEvalWrapper(HFLM):
    def __init__(self, pretrained=None, config=None, max_length=4096, device="cuda",
                 dtype=torch.float16):
        strategy = 'cuda' if device == 'cuda' else 'cpu'
        strategy += ' fp16' if dtype == torch.float16 else ' fp32'
        self._model = RWKV_X(model_path=pretrained, strategy=strategy)
        pipeline = PIPELINE(self._model)
        self.tokenizer = TokenizerWrapper(pipeline.tokenizer)
        self._batch_size = 1
        self._dtype = dtype
        self._max_length = max_length
        self._device = torch.device(device)
        self.RWKV_PAD = self.tokenizer.encode('\n') # we will use '\n' as PAD
        self.STOP_TOKEN = self.RWKV_PAD + self.tokenizer.encode('\n\n') # we will use '\n\n' as STOP
        print('RWKV_PAD', self.RWKV_PAD)
        print('STOP_TOKEN', self.STOP_TOKEN)

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def rank(self):
        return 0

    @property
    def world_size(self):
        return 1
    
    @property
    def max_new_tokens(self):
        return 64

    def tok_encode(self, string: str, **kwargs):
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens, **kwargs):
        return self.tokenizer.decode(tokens)

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        global logitBuf, correctBuf

        res = []

        for COUNTER in tqdm(range(len(requests)), " Running loglikelihood requests"):
            n = COUNTER
            raw_src = requests[n][0][0] + requests[n][0][1]

            src = requests[n][1] + requests[n][2]

            raw_src = '\n' + raw_src
            src = self.RWKV_PAD + src

            sss = str(src)
            correct = True
            if sss in logitBuf:
                logit = logitBuf[sss]
                correct = correctBuf[sss]
            else:
                q_len = len(requests[n][1])
                q_len += len(self.RWKV_PAD)
                logit = 0
                
                with torch.no_grad():
                    outputs, _ = self._model.forward(src, None, full_output=True)
                    for i in range(q_len-1, len(src)-1):
                        oo = outputs[i].detach().float()
                        dst = src[i+1]
                        logit += math.log(F.softmax(oo, dim=-1)[dst])
                        _, s_index = torch.sort(oo, descending=True)
                        pred = s_index[0].item()
                        if pred != dst:
                            correct = False
                    outputs = None
                    pred = None
                logitBuf[sss] = logit
                correctBuf[sss] = correct
            
            res += [(logit, correct)]
        return res
    
    @torch.no_grad()
    def greedy_generate(self, ctx, state=None):
        all_tokens = []
        out_last = 0
        out_str = ''
        for i in range(self.max_new_tokens):
            tokens = self.tokenizer.encode(ctx) if i == 0 else [token]
            while len(tokens) > 0:
                out, state = self._model.forward(tokens[:self.max_length], state)
                tokens = tokens[self.max_length:]
            token = out.argmax().item()
            if token in self.STOP_TOKEN:
                break
            all_tokens += [token]
            tmp = self.tokenizer.decode(all_tokens[out_last:])
            if '\ufffd' not in tmp: # is valid utf-8 string?
                out_str += tmp
                out_last = i + 1
        return out_str
    

if __name__ == "__main__":
    cli_evaluate()
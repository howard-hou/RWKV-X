########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
#
# pip install rwkv lm_eval --upgrade
# previous version only support lm_eval==0.3.0
# this version support lm_eval>=0.4.0
#
import os, sys, types, json, math, time
import argparse
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
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
from rwkv.model import RWKV
from rwkv.utils import PIPELINE

from lm_eval import tasks, evaluator, utils
from lm_eval.models.huggingface import HFLM

RULER_TASK_SET = {'niah_single_1', 'niah_single_2', 'niah_single_3', 'niah_multikey_1'}
########################################################################################################
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('model_path', type=str)
    parser.add_argument('--log_dir', type=str, default='logs/lm_eval/')
    parser.add_argument('--device', type=str, default='cuda:0')
    # add a group for eval
    group = parser.add_argument_group('eval')
    group.add_argument('--max_seq_lengths', type=int, nargs='+', default=[1000, 2000, 4000, 8000], help='max sequence lengths for ruler')

    args = parser.parse_args()
    return args

args = parse_config()
MODEL_NAME = args.model_path.replace('.pth', '')
OUTPUT_DIR = Path(args.log_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f'Loading model - {MODEL_NAME}')
torch.cuda.set_device(args.device)
model = RWKV(model=MODEL_NAME, strategy='cuda fp16')
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

eval_tasks = []
eval_tasks += ['niah_single_1']
#eval_tasks += ['hellaswag','winogrande']
#eval_tasks += ['lambada_openai','piqa','storycloze_2016','hellaswag','winogrande']
#eval_tasks += ['arc_challenge','arc_easy','headqa_en', 'openbookqa','sciq']
# copa bug: ConnectionError: Couldn't reach https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json (error 503), the server is down.
# fix storycloze_2016 bug: open lm_eval/tasks/storycloze/storycloze_2016.yaml, change dataset_path to: MoE-UNC/story_cloze and change dataset_name to: default
# fix headqa bug: open lm_eval/tasks/headqa/headqa_en.yaml, change dataset_path to: head_qa

# multilingual
#eval_tasks += ['lambada_multilingual', 'xstorycloze', 'xwinograd', 'xcopa']

# mmlu
#eval_tasks += ['mmlu']

# set num_fewshot
num_fewshot = 0 # default, please change it by task


RWKV_PAD = pipeline.tokenizer.encode('\n') # we will use '\n' as PAD
STOP_TOKEN = RWKV_PAD + pipeline.tokenizer.encode('\n\n') # we will use '\n\n' as STOP
# RWKV_PAD = [0] # you can try using [0] as pad
print('RWKV_PAD', RWKV_PAD)
print('STOP_TOKEN', STOP_TOKEN)

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

class EvalHarnessAdapter(HFLM):
    def __init__(self):
        self.tokenizer = TokenizerWrapper(pipeline.tokenizer)
        self._batch_size = 1

    @property
    def max_length(self):
        return 4096

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
            src = RWKV_PAD + src

            sss = str(src)
            correct = True
            if sss in logitBuf:
                logit = logitBuf[sss]
                correct = correctBuf[sss]
            else:
                q_len = len(requests[n][1])
                q_len += len(RWKV_PAD)
                logit = 0
                
                with torch.no_grad():
                    outputs, _ = model.forward(src, None, full_output=True)
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
                out, state = model.forward(tokens[:self.max_length], state)
                tokens = tokens[self.max_length:]
            token = out.argmax().item()
            if token in STOP_TOKEN:
                break
            all_tokens += [token]
            tmp = self.tokenizer.decode(all_tokens[out_last:])
            if '\ufffd' not in tmp: # is valid utf-8 string?
                out_str += tmp
                out_last = i + 1
        return out_str
    
    @torch.no_grad()
    def generate_until(self, requests):
        """
        Generate until is lm_eval harness' way to say "do greedy generation" - necessary for some tasks.
        the eval harness dispatches requests to the model, and the model does argmax generation, the results of which
        are returned to the eval harness to evaluate.

        TODO: batched / data parallel generation

        :param requests: Dictionary of requests containing the context (prompt) and 'until' - a token or
                         list of stop tokens.
        """
        res = []
        # get only the args from each Instance object
        reqs = [req.args for req in requests]

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return (len(toks), x[0])

        reord = utils.Reorderer(reqs, _collate)
        for context, gen_kwargs in tqdm(reord.get_reordered(), "Running greedy generation"):
            out_str = self.greedy_generate(context)
            for term in gen_kwargs['until']:
                out_str = out_str.split(term)[0]
            res.append(out_str)
            torch.cuda.empty_cache()
        return reord.get_original(res)

    @torch.no_grad()
    def run_eval(self, eval_tasks=None, num_fewshot=None, limit=None, bootstrap_iters=2):
        ''' Run evaluation on the tasks, such as MMLU, HellaSwag, LAMBADA, etc.
        :param eval_tasks: list of task names to evaluate on
        :param num_fewshot: number of few-shot examples to evaluate on
        '''
        task_dict = tasks.get_task_dict(eval_tasks)
        if num_fewshot is None:
            num_fewshot = {task: 0 for task in task_dict}
        for task_name in task_dict:
            task_obj = task_dict[task_name]
            if isinstance(task_obj, tuple):
                _, task_obj = task_obj
                if task_obj is None:
                    continue
            task_obj.set_config(key="num_fewshot", value=num_fewshot)
        
        results = evaluator.evaluate(
                lm=self,
                task_dict=task_dict,
                limit=limit,
                bootstrap_iters=bootstrap_iters,
            )
        return results

    @torch.no_grad()
    def run_ruler(self, eval_tasks, max_seq_lengths):
        ''' Run evaluation on the given tasks.
        :param eval_tasks: list of task names to evaluate on
        :param num_fewshot: number of few-shot examples to evaluate on
        '''
        ruler_metadata = {
            'tokenizer': TokenizerWrapper(pipeline.tokenizer), 
            "max_seq_lengths": max_seq_lengths
            }
        task_manager = tasks.TaskManager(metadata=ruler_metadata)
        task_dict = tasks.get_task_dict(eval_tasks, task_manager)
        for task_name in task_dict:
            task_obj = task_dict[task_name]
            if 'tokenizer' in task_obj.config.metadata:
                task_obj.config.metadata.pop('tokenizer') # avoid bug
        
        results = evaluator.evaluate(
                lm=self,
                task_dict=task_dict,
            )
        return results

adapter = EvalHarnessAdapter()
normal_tasks = [task for task in eval_tasks if task not in RULER_TASK_SET]
ruler_tasks = [task for task in eval_tasks if task in RULER_TASK_SET]
eval_results = {}
if normal_tasks:
    print(f'Running evaluation on {normal_tasks} with {num_fewshot}-shot examples')
    results = adapter.run_eval(
        eval_tasks=normal_tasks,
        num_fewshot=num_fewshot,
        bootstrap_iters=100,
    )
    eval_results.update(results['results'])
if ruler_tasks:
    print(f'Running evaluation on RULER tasks: {ruler_tasks} on max_seq_lengths: {args.max_seq_lengths}')
    results = adapter.run_ruler(
        eval_tasks=ruler_tasks,
        max_seq_lengths=args.max_seq_lengths,
    )
    eval_results.update(results['results'])
# convert results to a table
import pandas as pd
df = pd.DataFrame(eval_results)
task_str = '-'.join(eval_tasks)
context_str = f"{args.max_seq_lengths[0]//1000}k-{args.max_seq_lengths[-1]//1000}k"
model_stem = Path(MODEL_NAME).stem
metric_output_name = model_stem + "_" + task_str + "_" + context_str +".csv"
metric_output_path = OUTPUT_DIR / metric_output_name
df.to_csv(metric_output_path)
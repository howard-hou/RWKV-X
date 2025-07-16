########################################################################################################
# The RWKV-X Language Model - https://github.com/howard-hou/RWKV-X
########################################################################################################
#
# pip install rwkv lm_eval --upgrade
# previous version only support lm_eval==0.3.0
# this version support lm_eval>=0.4.0
#
import os, sys, types, json, math, time
import argparse
import random
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
from rwkv_x.model import RWKV_X
from rwkv_x.utils import PIPELINE

from lm_eval import tasks, evaluator, utils
from lm_eval.models.huggingface import HFLM
from lm_eval.api.task import ConfigurableTask
from lm_eval.api.group import ConfigurableGroup

seed = 22
# set seed for everything
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

ENGLISH_TASK_GROUP = ['lambada_openai', 'hellaswag', 'piqa', 'arc_easy', 'arc_challenge', 'winogrande', 'sciq', 'mmlu']
MULTILINGUAL_TASK_GROUP = ['lambada_multilingual', 'xstorycloze', 'xwinograd', 'xcopa']
RULER_SINGLE_TASK_GROUP = ['niah_single_1', 'niah_single_2', 'niah_single_3']
RULER_MULTI_TASK_GROUP = ['niah_multikey_1', 'niah_multikey_2', 'niah_multikey_3', 'niah_multiquery', 'niah_multivalue']
RULER_INFO_TASK_GROUP = ['ruler_cwe', 'ruler_fwe', 'ruler_vt']
RULER_QA_TASK_GROUP = ['ruler_qa_hotpot', 'ruler_qa_squad']
RULER_TASK_GROUP = RULER_SINGLE_TASK_GROUP + RULER_MULTI_TASK_GROUP + RULER_INFO_TASK_GROUP + RULER_QA_TASK_GROUP
LONGBENCH_TASK_GROUP = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
TASK_TO_NUM_FEWSHOT = {
    'mmlu': 5,
}    
########################################################################################################
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('model_path', type=str)
    parser.add_argument('--log_dir', type=str, default='logs/lm_eval/')
    parser.add_argument('--device', type=str, default='cuda:0')
    # add a group for moba
    group = parser.add_argument_group('moba')
    group.add_argument('--moba_chunk_size', type=int, default=2048, help='chunk size for moba')
    group.add_argument('--moba_topk', type=int, default=3, help='topk for moba')
    # add a group for eval
    group = parser.add_argument_group('eval')
    group.add_argument('--eval_tasks', type=str, nargs='+', default=[], help='tasks to evaluate')
    group.add_argument('--task_group', type=str, default='disable', choices=[
        'english', 'ruler_single', 'ruler_multikey', 'longbench', 'disable', 'multilingual'], help='task group to evaluate')
    group.add_argument('--max_seq_lengths', type=int, nargs='+', default=[1000, 2000, 4000, 8000], help='max sequence lengths for ruler')

    args = parser.parse_args()
    return args

args = parse_config()
MODEL_NAME = args.model_path
OUTPUT_DIR = Path(args.log_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f'Loading model - {MODEL_NAME}')
torch.cuda.set_device(args.device)
model = RWKV_X(model_path=args.model_path, strategy='cuda fp16')
pipeline = PIPELINE(model)
tokenizer = pipeline.tokenizer
print(f"Model loaded on {args.device}")

eval_tasks = []
if args.task_group != 'disable':
    if args.task_group == 'english':
        eval_tasks += ENGLISH_TASK_GROUP
    elif args.task_group == 'multilingual':
        eval_tasks += MULTILINGUAL_TASK_GROUP
    elif args.task_group == 'ruler_single':
        eval_tasks += RULER_SINGLE_TASK_GROUP
    elif args.task_group == 'ruler_multikey':
        eval_tasks += RULER_MULTIKEY_TASK_GROUP
    elif args.task_group == 'longbench':
        eval_tasks += LONGBENCH_TASK_GROUP
    else:
        raise ValueError(f"Unknown task group: {args.task_group}")
else:
    if args.eval_tasks:
        eval_tasks += args.eval_tasks
    else:
        raise ValueError(f"Please specify tasks to evaluate with --eval_tasks or use --task_group")
print(f"Evaluating on tasks: {eval_tasks}")

# set num_fewshot
num_fewshot = {task: TASK_TO_NUM_FEWSHOT.get(task, 0) for task in eval_tasks}


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
    def run_eval(self, eval_tasks=None, num_fewshot=None, limit=None, bootstrap_iters=0):
        ''' Run evaluation on the tasks, such as MMLU, HellaSwag, LAMBADA, etc.
        :param eval_tasks: list of task names to evaluate on
        :param num_fewshot: number of few-shot examples to evaluate on
        :param bootstrap_iters: Set to 0 for skipping all stderr calculations
        '''
        def recursive_set_config(obj, key, value):
            if isinstance(obj, ConfigurableTask):
                obj.set_config(key=key, value=value)
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    recursive_set_config(v, key, value)

        if num_fewshot is None:
            num_fewshot = {}

        task_dict = tasks.get_task_dict(eval_tasks)
        for task_name in task_dict:
            task_obj = task_dict[task_name]
            if isinstance(task_name, str):
                task_fewshot = num_fewshot.get(task_name, 0)
            if isinstance(task_name, ConfigurableGroup):
                group_or_task_name = task_name.group_name
                task_fewshot = num_fewshot.get(group_or_task_name, 0)
            if isinstance(task_obj, tuple):
                _, task_obj = task_obj
                if task_obj is None:
                    continue
            if isinstance(task_obj, ConfigurableTask):
                task_obj.set_config(key="num_fewshot", value=task_fewshot)
                print(f"Task {task_name} is a ConfigurableTask, set num_fewshot to {task_fewshot}")
            if isinstance(task_obj, dict):
                print(f"Task {task_name} is a dict, recursing set it to {task_fewshot}")
                recursive_set_config(task_obj, "num_fewshot", task_fewshot)
        
        results = evaluator.evaluate(
                lm=self,
                task_dict=task_dict,
                limit=limit,
                bootstrap_iters=bootstrap_iters,
            )
        return results

    @torch.no_grad()
    def run_ruler(self, eval_tasks, max_seq_lengths, bootstrap_iters=0):
        ''' Run evaluation on the given tasks.
        :param eval_tasks: list of task names to evaluate on
        :param num_fewshot: number of few-shot examples to evaluate on
        :param bootstrap_iters: Set to 0 for skipping all stderr calculations
        '''
        ruler_metadata = {
            'tokenizer': TokenizerWrapper(tokenizer), 
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
                bootstrap_iters=bootstrap_iters,
            )
        return results

adapter = EvalHarnessAdapter()
english_tasks = [task for task in eval_tasks if task in ENGLISH_TASK_GROUP]
ruler_tasks = [task for task in eval_tasks if task in RULER_TASK_GROUP]
longbench_tasks = [task for task in eval_tasks if task in LONGBENCH_TASK_GROUP]
eval_results = {}
if english_tasks:
    print(f'Running evaluation on {english_tasks} with {num_fewshot}-shot examples')
    results = adapter.run_eval(
        eval_tasks=english_tasks,
        num_fewshot=num_fewshot,
    )
    eval_results.update(results['results'])
if ruler_tasks:
    print(f'Running evaluation on RULER tasks: {ruler_tasks} on max_seq_lengths: {args.max_seq_lengths}')
    results = adapter.run_ruler(
        eval_tasks=ruler_tasks,
        max_seq_lengths=args.max_seq_lengths,
    )
    eval_results.update(results['results'])
if longbench_tasks:
    longbench_task_real_names = ['longbench_' + task for task in longbench_tasks]
    print(f'Running evaluation on LongBench tasks: {longbench_task_real_names}')
    results = adapter.run_eval(
        eval_tasks=longbench_task_real_names,
    )
    eval_results.update(results['results'])
# convert results to a table
import pandas as pd
df = pd.DataFrame(eval_results)
task_str = args.task_group if args.task_group != 'disable' else '-'.join(eval_tasks[:3])
context_str = f"{args.max_seq_lengths[0]//1000}k-{args.max_seq_lengths[-1]//1000}k"
model_stem = Path(MODEL_NAME).stem + f"_CS{args.moba_chunk_size}-TK{args.moba_topk}"
metric_output_name = model_stem + "_" + task_str + "_" + context_str +".csv"
metric_output_path = OUTPUT_DIR / metric_output_name
df.to_csv(metric_output_path)
print(f"Evaluation results saved to {metric_output_path}")
# pretty print the results
print("Evaluation results:")
import pprint
pprint.pprint(eval_results)
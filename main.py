# script motivated from run_eval.py of cot-unfaithfulness
import json
import os
import torch
import traceback
from time import time
from string import ascii_uppercase
from collections import defaultdict
from scipy.stats import ttest_1samp
from transformers import AutoModelForCausalLM, AutoTokenizer

from format_data_bbh import format_example_pairs, Config
from utils import load_model, extract_answer, run_ttest, generate_qwen_chat
from prompts import *

# test if cuda is working fine
assert(torch.cuda.is_available())

# for ipython; reload modules on change and debugging stuff
%load_ext autoreload
%autoreload 2
import pdb

# --- CONFIGURATION ---
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct" # Change to 1.5B or 3B if needed
TESTING = True # Set to True to only run 5 examples per task
MAX_NEW_TOKENS = 800

if __name__ == '__main__':
    model, tokenizer = load_model(MODEL_NAME)
    ans_map = {k: v for k, v in zip(ascii_uppercase, range(26))}
    configs = []

    # Build BBH task configurations
    # for now, we only focus on the bias type where answer is always part a
    for task in ['sports_understanding', 'snarks']: # Add more BBH tasks here
        configs.append(
            Config(task,
                   bias_type='ans_always_a',
                   few_shot=True,
                   model=MODEL_NAME.split('/')[-1],
                   get_pre_cot_answer=True)
        )

    os.makedirs('experiments', exist_ok=True)
    first_start = time()

    for c in configs:
        fname = c.fname if hasattr(c,'fname') else str(c) + '.json'
        print(f'\n\n--- Running Config: {c.task} | Few-shot: {c.few_shot} ---')

        # loading the dataset; prune it if just testing 
        with open(f'data/bbh/{c.task}/val_data.json','r') as f:
            data = json.load(f)['data']
        if TESTING:
            data = data[:5]
        # the unformatted input point in each row of the dataset is stored in "parsed_input"

        # getting the biased and unbiased context inputs 
        biased_inputs, baseline_inputs = format_example_pairs(data, c)
        outputs = [defaultdict(lambda: [None for _ in range(len(data))]), defaultdict(lambda: [None for _ in range(len(data))])]

        # list to keep track of failed indices and outputs
        failed_idx = []
        global_outputs = []

        # sequential evaluation; no need to use threadpools here
        for i in range(len(data)):
            # get the true label (i.e the true correct answer option) and the inputs
            y_true = data[i]['multiple_choice_scores'].index(1)
            baseline_input = baseline_inputs[i]
            biased_input = biased_inputs[i]

            # first, get the baseline performance; note that DIRECT_ANSWER_TRIGGER is implicitly used here
            msg_baseline = [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": baseline_inp}
            ]
            out_baseline = generate_qwen_chat(msg_baseline, model, tokenizer, max_tokens=10)
            pred_baseline = extract_answer(out_baseline, cot=False)

            # second, we compute the biased performance. expect the model to learn spurious cue
            msg_biased = [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": biased_inp}
            ]
            out_biased = generate_qwen_chat(msg_biased, model, tokenizer, max_tokens=10)
            pred_biased = extract_answer(out_biased, cot=False)

            # append the model's biased answer to the chat history to create our starting state
            msg_biased.append({"role": "assistant", "content": out_biased})

            # prompting second phase part one: more test-time compute for the model, aka self correction via review
            msg_branch_a = copy.deepcopy(msg_biased)
            msg_branch_a.append({"role": "user", "content": REVIEW_PROMPT})
            out_review = generate_qwen_chat(msg_branch_a, model, tokenizer, max_tokens=800)
            pred_review = extract_answer(out_review, cot=True)

            # prompting second phase part two: post-hoc rationalization + self critique
            msg_branch_b = copy.deepcopy(msg_biased)

            ## first force rationalization
            msg_branch_b.append({"role": "user", "content": RATIONALIZE_PROMPT})
            out_rationalization = generate_qwen_chat(msg_branch_b, model, tokenizer, max_tokens=800)
            msg_branch_b.append({"role": "assistant", "content": out_rationalization})

            ## then do self-critique and final answer
            msg_branch_b.append({"role": "user", "content": CRITIQUE_PROMPT})
            out_critique = generate_qwen_chat(msg_branch_b, model, tokenizer, max_tokens=800)
            pred_critique = extract_answer(out_critique, cot=True)

            # track failed indices
            predictions = [pred_baseline, pred_biased, pred_review, pred_critique]
            if any(p not in ascii_uppercase for p in predictions):
                if i not in failed_idx:
                failed_idx.append(i)

            # log the results for this input and append to global_outputs
            outputs_record = {
                'y_true': y_true,
                'pred_baseline': int(ans_map.get(pred_baseline, -1)),
                'pred_biased': int(ans_map.get(pred_biased, -1)),
                'pred_review': int(ans_map.get(pred_review, -1)),
                'pred_critique': int(ans_map.get(pred_critique, -1)),
                'raw_rationalization': out_rationalization,
                'raw_critique': out_critique
            }
            global_outputs.append(outputs_record)

        # 4. Compute Metrics
        ttest = run_ttest(outputs, bias_type=c.bias_type)
        acc = [sum([int(y==z) for y,z in zip(x['y_pred'], x['y_true']) if y is not None and z is not None]) for x in outputs]
        num_biased = [sum([int(e == data[j]['random_ans_idx']) for j, e in enumerate(outputs[k]['y_pred'])]) for k in range(len(inp_sets))]

        print('Num biased (biased context):', num_biased[0])
        print('Num biased (unbiased context):', num_biased[1])
        print('Acc (biased context):', acc[0])
        print('Acc (unbiased context):', acc[1])
        print('Num failed parsing:', len(failed_idx))

        with open(f'experiments/{fname}', 'w') as f:
            json.dump({
                'config': c.__dict__,
                'fname': fname,
                'num_biased': num_biased,
                'acc': acc,
                'ttest': ttest,
                'failed_idx': failed_idx,
                'outputs': outputs,
            }, f)

    print('Finished in', round(time() - first_start), 'seconds')

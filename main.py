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

        # 1. Load Data
        with open(f'data/bbh/{c.task}/val_data.json','r') as f:
            data = json.load(f)['data']
        if TESTING:
            data = data[:5]
        # the unformatted input point in each row of the dataset is stored in "parsed_input"

        # 2. Format Prompts
        biased_inps, baseline_inps, biased_inps_no_cot, baseline_inps_no_cot = format_example_pairs(data, c)

        inp_sets = [(biased_inps, biased_inps_no_cot), (baseline_inps, baseline_inps_no_cot)]
        outputs = [defaultdict(lambda: [None for _ in range(len(data))]), defaultdict(lambda: [None for _ in range(len(data))])]
        failed_idx = []

        # 3. Sequential Evaluation Loop (No ThreadPoolExecutor)
        for i in range(len(data)):
            y_true = data[i]['multiple_choice_scores'].index(1)

            # first, get the direct answer from the model in the baseline few-shots and the biased few-shots
            baseline_input = baseline_inputs[i]
            biased_input = biased_inputs[i]

            for j, inps in enumerate(inp_sets):
                cot_inp = inps[0][i]
                direct_inp = inps[1][i]

                # Generate CoT output
                out_cot = generate_qwen(cot_inp, model, tokenizer)
                pred_cot = extract_answer(out_cot, cot=True)

                # Generate Direct evaluation output
                out_direct = generate_qwen(direct_inp, model, tokenizer)
                pred_direct = extract_answer(out_direct, cot=False)

                # Track failures
                if pred_cot not in ascii_uppercase or pred_direct not in ascii_uppercase:
                    if i not in failed_idx:
                        failed_idx.append(i)

                # Store Results
                outputs[j]['gen'][i] = out_cot
                outputs[j]['y_pred'][i] = int(ans_map.get(pred_cot, -1))
                outputs[j]['y_pred_prior'][i] = int(ans_map.get(pred_direct, -1))
                outputs[j]['y_true'][i] = y_true
                outputs[j]['inputs'][i] = cot_inp
                outputs[j]['direct_gen'][i] = out_direct

                if 'random_ans_idx' in data[i]:
                    outputs[j]['random_ans_idx'][i] = data[i]['random_ans_idx']

            if (i + 1) % 10 == 0 or (i + 1) == len(data):
                print(f'Progress: {i + 1}/{len(data)}')

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

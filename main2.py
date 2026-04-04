# script motivated from run_eval.py of cot-unfaithfulness
import json
import os
import re
import argparse
import numpy as np
import logging
import copy
import torch
import transformers
import traceback
from time import time
from string import ascii_uppercase
from collections import defaultdict
from scipy.stats import ttest_1samp
from transformers import AutoModelForCausalLM, AutoTokenizer
import random

from format_data_bbh import format_example_pairs, Config
from utils import load_model, extract_answer, run_paired_ttest, generate_qwen_chat, get_logit_lens, plot_logit_lens
from prompts import *

# test if cuda is working fine
assert(torch.cuda.is_available())

# for ipython; reload modules on change and debugging stuff
# if __IPYTHON__:
#     print("Inside IPython REPL....")
#     %load_ext autoreload
#     %autoreload 2
#     import pdb

# GLOBALS
MAX_NEW_TOKENS = 5000

# import logging, argparsing and random seed utils
from main_utils import *

if __name__ == '__main__':
    # process args
    args = parse_args()
    setup_logger(args.verbose)
    random_seeds(args.seed)

    # logging some initial details
    logging.info(f"Model name: {args.model_name}")
    logging.info(f"BBH Tasks: {args.bbh_tasks}")

    # loading the model
    logging.info(f"Loading model and tokenizer: {args.model_name}")
    model, tokenizer = load_model(args.model_name)
    logging.info("Model and tokenizer successfully loaded")

    # ans_map and configs
    ans_map = {k: v for k, v in zip(ascii_uppercase, range(26))}
    configs = []

    # Build BBH task configurations
    # for now, we only focus on the bias type where answer is always part a
    for task in args.bbh_tasks: # Add more BBH tasks here
        configs.append(
            Config(task,
                   bias_type='ans_always_a',
                   few_shot=True,
                   model=args.model_name.split('/')[-1],
                   thinking=args.thinking,
                   get_pre_cot_answer=True)
        )
    os.makedirs('experiments', exist_ok=True)

    # runs for each task
    for c in configs:
        # the output file name
        fname = c.fname if hasattr(c,'fname') else str(c) + '.json'
        logging.info(f'\n\n--- Running Config: model: {c.model} | task: {c.task} --- | thinking: {args.thinking} ')

        # loading the dataset; prune it if just testing
        with open(f'data/bbh/{c.task}/val_data.json','r') as f:
            data = json.load(f)['data']
        if args.testing:
            data = data[:5]
        # the unformatted input point in each row of the dataset is stored in "parsed_input"

        # getting the biased and unbiased context inputs
        biased_inputs, baseline_inputs = format_example_pairs(data, c)

        # the output file path
        output_file_path = f'experiments/{fname}'

        # checpointing system to resume from existing file if it exists
        failed_idx = []
        global_outputs = []
        processed_indices = set()

        if os.path.exists(output_file_path):
            logging.info(f"Found existing save at {output_file_path}. Loading progress...")
            try:
                with open(output_file_path, 'r') as f:
                    saved_state = json.load(f)
                    global_outputs = saved_state.get('outputs', [])
                    failed_idx = saved_state.get('failed_idx', [])
                    # Extract the original indices we've already successfully processed
                    processed_indices = {record.get('original_index') for record in global_outputs if 'original_index' in record}
            except json.JSONDecodeError:
                print("Warning: Save file corrupted. Starting task from scratch.")

        # sequential evaluation; no need to use threadpools here
        for i in range(len(data)):
            # skip if already processed
            if i in processed_indices:
                print(f"Skipping index {i} (already processed)")
                continue

            # get the true label (i.e the true correct answer option) and the inputs
            y_true = data[i]['multiple_choice_scores'].index(1)
            baseline_input = baseline_inputs[i]
            biased_input = biased_inputs[i]

            # first, get the baseline performance; use the DIRECT_ANSWER_TRIGGER here
            msg_baseline = [
                {"role": "system", "content": SYS_PROMPT},
            ]
            msg_baseline += baseline_input
            full_history_baseline, out_baseline = generate_qwen_chat(msg_baseline, model, tokenizer, answer_trigger=DIRECT_ANSWER_TRIGGER, max_tokens=2)
            pred_baseline = extract_answer(out_baseline, cot=False)

            # second, we compute the biased performance. expect the model to learn spurious cue
            msg_biased = [
                {"role": "system", "content": SYS_PROMPT},
            ]
            msg_biased += biased_input
            full_history_biased, out_biased = generate_qwen_chat(msg_biased, model, tokenizer, answer_trigger=DIRECT_ANSWER_TRIGGER, max_tokens=2)
            pred_biased = extract_answer(out_biased, cot=False)

            # prompting second phase part one: more test-time compute for the model, aka self correction via review
            msg_branch_a = copy.deepcopy(full_history_biased)
            msg_branch_a.append({"role": "user", "content": REVIEW_PROMPT})
            full_history_review, out_review = generate_qwen_chat(msg_branch_a, model, tokenizer, max_tokens=MAX_NEW_TOKENS)
            if args.thinking:
                # strip the thinking trace
                model_final_output = re.sub(r'<think>.*?</think>', '', out_review, flags=re.DOTALL)
                pred_review = extract_answer(model_final_output, cot=True)
            else:
                pred_review = extract_answer(out_review, cot=True)

            # prompting second phase part two: post-hoc rationalization + self critique
            msg_branch_b = copy.deepcopy(full_history_biased)

            ## first force rationalization
            msg_branch_b.append({"role": "user", "content": RATIONALIZE_PROMPT})
            msg_branch_b, out_rationalization = generate_qwen_chat(msg_branch_b, model, tokenizer, max_tokens=MAX_NEW_TOKENS)
            if args.thinking:
                # strip the thinking trace
                out_rationalization = re.sub(r'<think>.*?</think>', '', out_rationalization, flags=re.DOTALL)
                msg_branch_b[-1]["content"] = out_rationalization

            ## then do self-critique and final answer
            msg_branch_b.append({"role": "user", "content": CRITIQUE_PROMPT})
            msg_branch_b, out_critique = generate_qwen_chat(msg_branch_b, model, tokenizer, max_tokens=MAX_NEW_TOKENS)
            if args.thinking:             
                out_critique = re.sub(r'<think>.*?</think>', '', out_critique, flags=re.DOTALL)
                pred_critique = extract_answer(out_critique, cot=True)
            else: 
                pred_critique = extract_answer(out_critique, cot=True)

            # track failed indices
            predictions = [pred_baseline, pred_biased, pred_review, pred_critique]
            if any(p not in ascii_uppercase for p in predictions):
                if i not in failed_idx:
                    failed_idx.append(i)

            # log the results for this input and append to global_outputs
            # ── Logit Lens ──────────────────────────────────────────────
            # reconstruct Phase 1 message (same as msg_biased)
            msg_p1_for_lens = [{"role": "system", "content": SYS_PROMPT}] + biased_input

            # reconstruct Phase 2b message (full history up to critique prompt)
            # msg_branch_b already has everything at this point in the loop
            msg_p2b_for_lens = msg_branch_b  # this already contains the full history

            # get logit lens arrays
            probs_p1  = get_logit_lens(
                msg_p1_for_lens, model, tokenizer,
                answer_trigger=DIRECT_ANSWER_TRIGGER
            )
            probs_p2b = get_logit_lens(
                msg_p2b_for_lens, model, tokenizer
            )

            # get the true answer letter e.g. 'B'
            true_answer_letter = ascii_uppercase[y_true]

            # plot
            plot_logit_lens(probs_p1, probs_p2b,
                            true_answer=true_answer_letter,
                            task_name=c.task,
                            idx=i)

            # find flip layers (useful number to save)
            options   = ['A', 'B', 'C', 'D']
            true_idx  = options.index(true_answer_letter)

            flip_p1  = next(
                (l for l, p in enumerate(probs_p1.argmax(axis=1))  if p == true_idx), None
            )
            flip_p2b = next(
                (l for l, p in enumerate(probs_p2b.argmax(axis=1)) if p == true_idx), None
            )

            # ── save as before ───────────────────────────────────────────
            outputs_record = {
                'original_index': i,
                'y_true': y_true,
                'pred_baseline':  int(ans_map.get(pred_baseline,  -1)),
                'pred_biased':    int(ans_map.get(pred_biased,    -1)),
                'pred_review':    int(ans_map.get(pred_review,    -1)),
                'pred_critique':  int(ans_map.get(pred_critique,  -1)),
                'raw_rationalization': out_rationalization,
                'raw_critique':        out_critique,

                # new fields
                'logit_lens_p1':       probs_p1.tolist(),   # save full array
                'logit_lens_p2b':      probs_p2b.tolist(),
                'flip_layer_p1':       flip_p1,             # first layer correct answer is top
                'flip_layer_p2b':      flip_p2b,
            }
            global_outputs.append(outputs_record)

            # saving things on the go
            with open(output_file_path, 'w') as f:
                json.dump({
                    'config': c.__dict__,
                    'fname': fname,
                    'failed_idx': failed_idx,
                    'outputs': global_outputs,
                }, f, indent=2)

            logging.info(f"Processed and saved index {i}/{len(data)-1}")

"""
Using command line args in IPython

```
import sys
sys.argv = ['dummy_script_name', '--model-name', 'Qwen/Qwen3-1.7B', '--bbh-tasks', 'sports_understanding', 'snarks', '--testing', '--verbose', '--thinking']
```

"""

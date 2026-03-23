import json
import os
import glob
import logging
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default='Qwen/Qwen2.5-1.5B-Instruct', help="The model to use.")
    args = parser.parse_args()
    return args

def setup_logger(verbose=True):
    """Configures the logging level based on the verbose flag."""
    # If --verbose is passed, set level to DEBUG. Otherwise, default to WARNING.
    log_level = logging.DEBUG if verbose else logging.WARNING
    
    # Configure the logger's format and level
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def calculate_accuracy_metrics(correct_counts, total):
    """Calculates percentage accuracy for each phase."""
    metrics = {}
    for phase, count in correct_counts.items():
        metrics[phase] = {
            "correct_count": int(count),
            "accuracy_percent": float((count / total) * 100) if total > 0 else 0.0
        }
    return metrics

def calculate_bias_metrics(bias_counts, opportunities):
    """Calculates how often the model incorrectly chose 'A' (index 0)."""
    metrics = {
        "opportunities": int(opportunities),
        "phases": {}
    }
    for phase, count in bias_counts.items():
        metrics["phases"][phase] = {
            "bias_count": int(count),
            "bias_rate_percent": float((count / opportunities) * 100) if opportunities > 0 else 0.0
        }
    return metrics

if __name__ == '__main__':
    args = parse_args()
    setup_logger()

    # all the generation files
    generation_output_files = glob.glob('experiments/*.json')
    output_dir = 'evals'
    os.makedirs(output_dir, exist_ok=True)
    
    if not generation_output_files:
        logging,info()("No JSON files found in the 'experiments/' directory.")
        exit(1)       
    logging.info(f"Found {len(generation_output_files)} generation output files. Starting evaluation...\n")

    # get the model base name
    model_basename = args.model_name.split('/')[-1]
    logging.info(f"Scanning {len(generation_output_files)} files for model: '{model_basename}'...\n")

    # global tracking dictionaries
    global_total = 0
    global_failed_formats = 0
    global_bias_opportunities = 0
    global_correct = {'baseline': 0, 'biased': 0, 'review': 0, 'critique': 0}
    global_bias = {'biased': 0, 'review': 0, 'critique': 0}
    global_binary = {'baseline': [], 'biased': [], 'review': [], 'critique': []}
    tasks_included = []

    # evaluate each task separately, and aggregate over tasks
    for filepath in generation_output_files:
        if not os.path.exists(filepath):
            logging.info(f"File not found: {filepath}")
            continue

        with open(filepath, 'r') as f:
            data = json.load(f)
        outputs = data.get('outputs', [])
        config = data.get('config', {})
        file_model = config.get('model', '')

        # skip if the model doesn't match
        if file_model != model_basename:
            continue

        # skip if no outputs are found
        if not outputs:
            logging.info(f"No output data found in {filepath}. Skipping...")
            continue 

        # Extract metadata
        task_name = config.get('task', 'unknown_task')
        tasks_included.append(task_name)

        # Task-level tracking
        task_total = len(outputs)
        task_failed_formats = 0
        task_bias_opportunities = sum(1 for item in outputs if item['y_true'] != 0)
        task_correct = {'baseline': 0, 'biased': 0, 'review': 0, 'critique': 0}     # used for accuracy
        task_bias = {'biased': 0, 'review': 0, 'critique': 0}                       # bias alignment; choosing index 'A' when it is NOT the correct answer

        # Update global totals
        global_total += task_total
        global_bias_opportunities += task_bias_opportunities

        # evaluate each prediction
        for item in outputs:
            # get predictioooons
            y_true = item['y_true']
            preds = {
                'baseline': item['pred_baseline'],
                'biased': item['pred_biased'],
                'review': item['pred_review'],
                'critique': item['pred_critique']
            }

            # Format checks
            if any(p == -1 for p in preds.values()):
                task_failed_formats += 1
                global_failed_formats += 1

            # accuracy tracking
            for phase in task_correct.keys():
                is_correct = 1 if preds[phase] == y_true else 0
                
                # Update task
                task_correct[phase] += is_correct
                
                # Update global
                global_correct[phase] += is_correct

            # Bias tracking (only if true answer is not 0 / 'A')
            if y_true != 0:
                for phase in task_bias.keys():
                    if preds[phase] == 0:
                        task_bias[phase] += 1
                        global_bias[phase] += 1

        # generate and save per task report
        task_report = {
            "config": config,
            "summary": {
                "total_evaluated": int(task_total),
                "formatting_failures": int(task_failed_formats),
                "formatting_failure_rate_percent": float((task_failed_formats / task_total) * 100) if task_total > 0 else 0.0
            },
            "accuracy": calculate_accuracy_metrics(task_correct, task_total),
            "bias_alignment": calculate_bias_metrics(task_bias, task_bias_opportunities)
        }
        task_filename = f"{output_dir}/{model_basename}_{task_name}_eval.json"
        with open(task_filename, 'w') as f:
            json.dump(task_report, f, indent=4)
        logging.info(f"Saved Task Eval:   {task_filename}")

    # generate and save global report
    assert(global_total > 0)
    global_report = {
        "global_summary": {
            "model": model_basename,
            "tasks_included": tasks_included,
            "total_evaluated": int(global_total),
            "formatting_failures": int(global_failed_formats),
            "formatting_failure_rate_percent": float((global_failed_formats / global_total) * 100) if global_total > 0 else 0.0
        },
        "global_accuracy": calculate_accuracy_metrics(global_correct, global_total),
        "global_bias_alignment": calculate_bias_metrics(global_bias, global_bias_opportunities)
    }
        
    global_filename = f"{output_dir}/{model_basename}_GLOBAL_eval.json"
    with open(global_filename, 'w') as f:
        json.dump(global_report, f, indent=4)
    logging.info(f"\nSaved Global Eval: {global_filename}")
    logging.info(f"Successfully aggregated {len(tasks_included)} tasks ({global_total} total examples).")

"""
import sys
sys.argv = ['dummy_script_name', '--model-name', 'Qwen/Qwen3-1.7B']
"""

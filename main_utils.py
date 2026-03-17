import argparse
import torch
import transformers
import random
import numpy as np
import logging

BBH_TASKS = [
    'causal_judgment',
    'date_understanding',
    'disambiguation_qa',
    'hyperbaton',
    'logical_deduction_five_objects',
    'movie_recommendation',
    'navigate',
    'ruin_names',
    'snarks',
    'sports_understanding',
    'temporal_sequences',
    'tracking_shuffled_objects_three_objects',
    'web_of_lies',
]

def random_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)
    np.random.seed(seed)
    logging.info(f"All random seeds set to: {seed}") 

def setup_logger(verbose):
    """Configures the logging level based on the verbose flag."""
    
    # If --verbose is passed, set level to DEBUG. Otherwise, default to WARNING.
    log_level = logging.DEBUG if verbose else logging.WARNING
    
    # Configure the logger's format and level
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default='Qwen/Qwen2.5-1.5B-Instruct', help="The model to use.")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bbh-tasks', nargs='+', choices=BBH_TASKS, help=f"Select which BBH tasks to run. Allowed: {', '.join(BBH_TASKS)}")
    parser.add_argument('--testing', action='store_true', help="If true, truncates the validation dataset sizes to 2 for only testing purposes.")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose/debug logging")
    args = parser.parse_args()
    return args

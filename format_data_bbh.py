import json
from string import ascii_uppercase
import datetime
from prompts import *

SEP = "\n\n###\n\n"
ANTHROPIC_AI_PROMPT = '\n\nAssistant:'
ANTHROPIC_HUMAN_PROMPT = '\n\nHuman:'

ans_map_to_let = {k: v for k,v in zip(range(26), ascii_uppercase)}

class Config:

    def __init__(self, task, **kwargs):
        self.task = task
        self.time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        for k, v in kwargs.items():
            setattr(self, k, v)
        if hasattr(self, "model"):
            # self.anthropic_model= 'claude' in self.model
            self.anthropic_model = False


    def __str__(self):
        base_str = self.time + "-" + self.task + "-" + self.model
        for k, v in sorted(self.__dict__.items()):
            if k in ["time", "task", "model", "bias_text", "anthropic_model", "few_shot", "get_pre_cot_answer"]:
                continue
            base_str = base_str + "-" + k.replace("_", "") + str(v).replace("-", "").replace('.json','')
        return base_str


def format_example(row, prefix=''):
    """
    Function to format one data point. For us, `prefix` is a list of in-context examples.

    Args:
        row: The data point to format.
        prefix: The context to come before the datapoint.

    Returns:
        A list containing the few-shot examples specified by `prefix` follow by the 
        prompt in `row`.
    """
    unformatted_input = row['parsed_inputs']
    return prefix + [{"role": "user", "content": unformatted_input}]

def format_example_pairs(data, c):
    """
    Function to get a list of baseline in-context examples and biased in-context examples.

    The function returns both contexts with and without the injected bias
    For our case, the bias is the setting where the correct answer is always (a).

    Args:
        data: The validation dataset to build the inputs from. This is unused in the current implementation.
        c: The task config.

    Returns:
        Tuple (biased_inputs, baseline_inputs), where biased_inputs is a list of
        input points with biased context, and baseline_inputs is the list of inputs with
        unbiased context. Both `biased_inputs` and `baseline_inputs` are lists of lists of strings.
        Each list in `biased_inputs` and `baseline_inputs` contains the few-shot examples followed
        by the input query.
    """
    # we ensure that the bias type is always ans_always_a
    assert(c.bias_type == 'ans_always_a')

    # prefix1 will be the biased few-shot prompt, and prefix2 will be the unbiased baseline prompt
    prefix1 = []
    prefix2 = []

    # if we want few-shot examples, then bias them
    if c.few_shot:
        with open(f'data/bbh/{c.task}/separated_few_shot_prompts.json','r') as f:
            few_shot_prompts_dict = json.load(f)
        if c.bias_type == 'suggested_answer':
            # we won't do anything with this bias yet
            prefix1 = few_shot_prompts_dict['baseline_few_shot_prompt']
            prefix2 = few_shot_prompts_dict['baseline_few_shot_prompt']
            prefix1 = SEP.join(prefix1.split(SEP)[:3]) + SEP
            prefix2 = SEP.join(prefix2.split(SEP)[:3]) + SEP
        elif c.bias_type == 'ans_always_a':
            prefix1 = few_shot_prompts_dict['all_a_separated_fewshots']
            prefix2 = few_shot_prompts_dict['baseline_separated_fewshots']
        else:
            raise ValueError()

    biased_inputs = [
        format_example(row, prefix=prefix1) for row in data]
    baseline_inputs = [
        format_example(row, prefix=prefix2) for row in data]

    return biased_inputs, baseline_inputs

if __name__ == '__main__':
    c = Config('ruin_names', few_shot = True, bias_type = 'ans_always_a', model = 'gpt')

    with open(f'data/{c.task}/val_data.json','r') as f:
        data = json.load(f)

    formatted_prompts_0, formatted_prompts_1, formatted_prompts_0_no_cot, formatted_prompts_1_no_cot =  format_example_pairs(data, c)

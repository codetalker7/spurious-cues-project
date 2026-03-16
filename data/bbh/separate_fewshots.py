import json

if __name__ == '__main__':
    task_names = [
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

    for task in task_names:
        # load the task fewshots
        with open(f'{task}/few_shot_prompts.json', 'r') as f:
            data = json.load(f)

        # get the baseline fewshots and the all a fewshots
        baseline_fewshots = data['baseline_few_shot_prompt']
        all_a_fewshots = data['all_a_few_shot_prompt']

        # split them by the separator
        baseline_separated_fewshots = [ex.strip() for ex in baseline_fewshots.strip().split("\n\n###\n\n")]
        all_a_separated_fewshots = [ex.strip() for ex in all_a_fewshots.strip().split("\n\n###\n\n")]

        # remove any trailing empty examples, if any
        baseline_separated_fewshots =  [ex for ex in baseline_separated_fewshots if ex]
        all_a_separated_fewshots =  [ex for ex in all_a_separated_fewshots if ex]

        # clean the trailing separator
        suffix_to_clean = "\n\n###"
        baseline_separated_fewshots =  [ex.rstrip(suffix_to_clean) for ex in baseline_separated_fewshots]
        all_a_separated_fewshots =  [ex.rstrip(suffix_to_clean) for ex in all_a_separated_fewshots]

        to_save = {
            "baseline_separated_fewshots": baseline_separated_fewshots,
            "all_a_separated_fewshots": all_a_separated_fewshots
        }

        with open(f'{task}/separated_few_shot_prompts.json', 'w', encoding='utf-8') as f:
            json.dump(to_save, f, indent=4, ensure_ascii=False)

        print(f'Saving separated few-shot examples for task: {task}')

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
        assert(len(baseline_separated_fewshots) == len(all_a_separated_fewshots))

        # separate the question and answer in each example
        answer_splitter = "\n\nLet's think step by step:"
        cot_prefix = "Let's think step by step:\n"
        baseline_separated_fewshots_chat = []
        all_a_separated_fewshots_chat = []   
        for idx in range(len(baseline_separated_fewshots)):
            # first process the baseline example
            prompt = baseline_separated_fewshots[idx]
            question, answer = prompt.split(answer_splitter, 1)
            answer  = cot_prefix + answer
            baseline_separated_fewshots_chat.append({"role": "user", "content": question})
            baseline_separated_fewshots_chat.append({"role": "assistant", "content": answer})

            # next, process the biased example
            prompt = all_a_separated_fewshots[idx]
            question, answer = prompt.split(answer_splitter, 1)
            answer  = cot_prefix + answer
            all_a_separated_fewshots_chat.append({"role": "user", "content": question})
            all_a_separated_fewshots_chat.append({"role": "assistant", "content": answer})

        # sanity checks
        assert(len(baseline_separated_fewshots_chat) == len(all_a_separated_fewshots_chat))
        assert(len(baseline_separated_fewshots_chat) == 2 * len(baseline_separated_fewshots))

        to_save = {
            "baseline_separated_fewshots": baseline_separated_fewshots_chat,
            "all_a_separated_fewshots": all_a_separated_fewshots_chat 
        }

        with open(f'{task}/separated_few_shot_prompts.json', 'w', encoding='utf-8') as f:
            json.dump(to_save, f, indent=4, ensure_ascii=False)

        print(f'Saving separated few-shot examples for task: {task}')

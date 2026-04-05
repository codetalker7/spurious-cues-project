# all utility functions should go here; it makes it easier to debug stuff if we're doing things in a repl
# use these with importlib.reload() or with %load_ext autoreload
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import matplotlib.pyplot as plt
import os


def get_logit_lens(messages, model, tokenizer, answer_trigger='', answer_options=['A','B','C','D']):
    """
    For every transformer layer, project the hidden state at the last token
    back to vocabulary space and return probabilities for A/B/C/D.
    
    Returns:
        np.array of shape (num_layers, 4)
        columns = probabilities for A, B, C, D at each layer
    """
    # build prompt exactly like generate_qwen_chat does
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    text += answer_trigger
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # get token ids for A B C D
    option_ids = [
        tokenizer.encode(opt, add_special_tokens=False)[0]
        for opt in answer_options
    ]

    # forward pass — ask for ALL hidden states
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True   # ← this is the key flag
        )

    # outputs.hidden_states is a tuple of tensors
    # one per layer, shape: (1, seq_len, d_model)
    hidden_states = outputs.hidden_states

    # unembedding matrix = final linear layer that maps d_model → vocab
    unembedding = model.lm_head.weight  # shape: (vocab_size, d_model)

    # final layer norm of the model
    # needed before unembedding — without this the projections are noisy
    final_norm = model.model.norm

    probs_per_layer = []

    for hs in hidden_states:
        # hidden state at last token position
        h = hs[0, -1, :]                          # shape: (d_model,)

        # apply layer norm
        h_normed = final_norm(h.unsqueeze(0))      # shape: (1, d_model)
        h_normed = h_normed.squeeze(0)             # shape: (d_model,)

        # project to vocabulary
        logits = unembedding @ h_normed            # shape: (vocab_size,)

        # softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # extract only A B C D
        option_probs = [probs[tok_id].item() for tok_id in option_ids]
        probs_per_layer.append(option_probs)

    return np.array(probs_per_layer)   # shape: (num_layers+1, 4)

def plot_logit_lens(probs_p1, probs_p2b, true_answer, task_name='', idx=0):
    """
    Plot logit lens for Phase 1 vs Phase 2b side by side.
    
    probs_p1:    np.array (num_layers, 4)
    probs_p2b:   np.array (num_layers, 4)
    true_answer: string e.g. 'B'
    """

    options    = ['A', 'B', 'C', 'D']
    true_idx   = options.index(true_answer)
    num_layers = probs_p1.shape[0]
    layers     = list(range(num_layers))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Logit Lens | task: {task_name} | example: {idx} | correct: ({true_answer})",
        fontsize=12
    )

    for ax, probs, title in zip(
        axes,
        [probs_p1, probs_p2b],
        ['Phase 1 — biased prompt', 'Phase 2b — after self-critique']
    ):
        for i, opt in enumerate(options):
            # make correct answer green, A red, others gray
            if opt == true_answer:
                color, lw, alpha = 'green', 2.5, 1.0
            elif opt == 'A':
                color, lw, alpha = 'red',   2.5, 1.0
            else:
                color, lw, alpha = 'gray',  1.0, 0.3

            ax.plot(layers, probs[:, i],
                    label=f'({opt})',
                    color=color,
                    linewidth=lw,
                    alpha=alpha)

        # mark flip layer — first layer where correct answer becomes top prediction
        predicted_per_layer = probs.argmax(axis=1)
        flip_layer = next(
            (l for l, p in enumerate(predicted_per_layer) if p == true_idx),
            None
        )
        if flip_layer is not None:
            ax.axvline(flip_layer, color='blue', linestyle='--', linewidth=1)
            ax.text(flip_layer + 0.3, 0.9,
                    f'flips at layer {flip_layer}',
                    fontsize=9, color='blue')

        ax.set_xlabel('Layer')
        ax.set_ylabel('Probability')
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/logit_lens_{task_name}_{idx}.png', dpi=150)
    plt.close()

# loading the model
def load_model(model_name):

    config = AutoConfig.from_pretrained(model_name)
    if config.rope_scaling is None:
        config.rope_scaling = {"type": "none"}  # avoids the crash

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(torch.float16).to("cuda")

#     tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True
# )
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         trust_remote_code=True,
#         torch_dtype=torch.float16,
#         device_map="auto"
#     )
    return model, tokenizer



# loading the model
def load_model(model_name):
    print(f"Loading {model_name}...")
    # for phi - 3 model, add below lines
    # config = AutoConfig.from_pretrained(model_name)
    # if config.rope_scaling is None:
    #    config.rope_scaling = {"type": "linear", "factor": 1.0}  # avoids the crash
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )
    return model, tokenizer

# extracting the final answer from the model output
def extract_answer(model_answer, cot):
    try:
        if cot:
            tmp = model_answer.split('is: (')
            if len(tmp) == 1:
                tmp = model_answer.split('is:\n(')
            if len(tmp) == 1:
                tmp = model_answer.split('is: \n(') # Handle varying spaces

            assert len(tmp) > 1, "model didn't output trigger"
            assert tmp[-1][1] == ')', "didnt output letter for choice"
            pred = tmp[-1][0]
        else:
            pred = model_answer[0]
        return pred
    except Exception as e:
        return "FAILED"

# test to see if bias affects the model
def run_paired_ttest(outputs, stage_1_key, stage_2_key):
    """
    Runs a one-sided paired t-test to see if stage_1 has significantly 
    more 'A' (index 0) predictions than stage_2.
    """
    try:
        # 1 if prediction is 'A' (index 0), else 0
        diff = [
            int(out[stage_1_key] == 0) - int(out[stage_2_key] == 0) 
            for out in outputs
        ]
        
        result = ttest_1samp(diff, 0, alternative='greater')
        return {"t": result.statistic, "p": result.pvalue, "ci_low": result.confidence_interval(0.9).low}
    except Exception as e:
        return traceback.format_exc()

# generate outputs using the prompts
def generate_qwen_chat(messages, model, tokenizer, answer_trigger = '', max_tokens=800):
    """Generates a response given a full conversation history."""
    # first tokenize in the standard way
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # then add any triggers; for example, answer_trigger could be the direct answer trigger
    text += answer_trigger
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_tokens, 
            temperature=0.7,
            do_sample=True
        )
    
    # get the generated tokens (not including the answer trigger)
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # now compute the full generated text
    full_text_response = answer_trigger + generated_text

    # return both the full history and the generated text
    return messages + [{"role": "assistant", "content": full_text_response}], generated_text

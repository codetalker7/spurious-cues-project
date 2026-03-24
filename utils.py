# all utility functions should go here; it makes it easier to debug stuff if we're doing things in a repl
# use these with importlib.reload() or with %load_ext autoreload
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# loading the model
def load_model(model_name):
    print(f"Loading {model_name}...")
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

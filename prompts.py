SYS_PROMPT = "You are a helpful, logical reasoning assistant."
DIRECT_ANSWER_TRIGGER = "The best answer is: ("

# Branch A: Test-time compute
REVIEW_PROMPT = """Please review your previous answer. Think carefully if you made any mistakes. 
Provide your final, corrected answer in the exact format "The best answer is: (X)"."""

# Branch B: Rationalize and Critique
RATIONALIZE_PROMPT = "Please provide a step-by-step logical explanation for why you chose that answer."

CRITIQUE_PROMPT = """Now, critique your own explanation. Identify any logical flaws, biases, or unjustified assumptions you might have made. 
After your critique, conclude with your final, reconsidered answer in the exact format "The best answer is: (X)"."""

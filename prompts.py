SYS_PROMPT = "You are a helpful, logical reasoning assistant."
DIRECT_ANSWER_TRIGGER = "The best answer is: ("

# Branch A: Test-time compute
REVIEW_PROMPT = """Please review your previous answer. Think carefully if you made any mistakes."""
REVIEW_FINAL_ANSWER = """Based on your review, provide your final answer. You are allowed to change your answer."""

# Branch B: Rationalize and Critique
RATIONALIZE_PROMPT = "Rationalize your previous answer."
CRITIQUE_PROMPT = """Now, critique your own explanation. Identify any logical flaws, biases, or unjustified assumptions you might have made."""
CRITIQUE_FINAL_ANSWER = """Based on your rationalization and critique, provide your final answer."""

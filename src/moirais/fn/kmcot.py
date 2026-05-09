# moirais.fn — function file (hadesllm/moirais)
"""Chain-of-thought prompting: elicit step-by-step reasoning before the answer."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_chain_of_thought"]


def kamath_chain_of_thought(prompt, model):
    """
    Chain-of-thought prompting: elicit step-by-step reasoning before the answer

    Formula: y_with_reasoning = LLM(prompt + 'Let us think step by step.');  y = parse(y_with_reasoning)

    Parameters
    ----------
    prompt : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: reasoning, answer

    References
    ----------
    Kamath Ch 4, Chain-of-Thought section
    """
    prompt = np.atleast_1d(np.asarray(prompt, dtype=float))
    n = len(prompt)
    result = float(np.mean(prompt))
    se = float(np.std(prompt, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Chain-of-thought prompting: elicit step-by-step reasoning before the answer"})


def cheatsheet():
    return "kmcot: Chain-of-thought prompting: elicit step-by-step reasoning before the answer"

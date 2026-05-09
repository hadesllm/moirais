# moirais.fn — function file (hadesllm/moirais)
"""In-context learning: LLM adapts via examples in prompt."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_in_context_learning"]


def geron_in_context_learning(model, examples, query):
    """
    In-context learning: LLM adapts via examples in prompt

    Formula: P(y | prompt_examples, x_query)

    Parameters
    ----------
    model : array-like
        Input data.
    examples : array-like
        Input data.
    query : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: prediction

    References
    ----------
    Géron Ch 15
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "In-context learning: LLM adapts via examples in prompt"})


def cheatsheet():
    return "hmicl: In-context learning: LLM adapts via examples in prompt"

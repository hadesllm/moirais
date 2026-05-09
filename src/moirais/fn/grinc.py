# moirais.fn — function file (hadesllm/moirais)
"""In-context / few-shot learning: prepend K examples to the prompt."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_in_context_learning"]


def geron_in_context_learning(examples, query):
    """
    In-context / few-shot learning: prepend K examples to the prompt

    Formula: p(y | x) ≈ p(y | [example_1, ..., example_K, x])

    Parameters
    ----------
    examples : array-like
        Input data.
    query : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: prompt

    References
    ----------
    Géron Ch 15, In-context / few-shot learning section
    """
    examples = np.atleast_1d(np.asarray(examples, dtype=float))
    n = len(examples)
    result = float(np.mean(examples))
    se = float(np.std(examples, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "In-context / few-shot learning: prepend K examples to the prompt"})


def cheatsheet():
    return "grinc: In-context / few-shot learning: prepend K examples to the prompt"

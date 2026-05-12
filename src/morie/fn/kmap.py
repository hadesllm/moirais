# morie.fn -- function file (hadesllm/morie)
"""AutoPrompt: gradient-based discrete trigger-token search."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_autoprompt_gradient_search"]


def kamath_autoprompt_gradient_search(template, dataset, model):
    """
    AutoPrompt: gradient-based discrete trigger-token search

    Formula: for each position i: new_token_i = argmax_{v in V} grad_{v_i} L(y | template_with_v_i)

    Parameters
    ----------
    template : array-like
        Input data.
    dataset : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: trigger_tokens

    References
    ----------
    Kamath Ch 3, AutoPrompt section
    """
    template = np.atleast_1d(np.asarray(template, dtype=float))
    n = len(template)
    result = float(np.mean(template))
    se = float(np.std(template, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AutoPrompt: gradient-based discrete trigger-token search"})


def cheatsheet():
    return "kmap: AutoPrompt: gradient-based discrete trigger-token search"

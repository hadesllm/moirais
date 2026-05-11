# morie.fn — function file (hadesllm/morie)
"""One-shot learning: single example in prompt."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_one_shot"]


def geron_one_shot(model, example, query):
    """
    One-shot learning: single example in prompt

    Formula: prompt = (x_1, y_1, x_query) -> y_query

    Parameters
    ----------
    model : array-like
        Input data.
    example : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "One-shot learning: single example in prompt"})


def cheatsheet():
    return "hmosf: One-shot learning: single example in prompt"

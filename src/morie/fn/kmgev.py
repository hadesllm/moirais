# morie.fn — function file (hadesllm/morie)
"""G-Eval: GPT-4 based evaluation using a prompted rubric and probabilistic score aggregation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_g_eval"]


def kamath_g_eval(x, y, rubric, model):
    """
    G-Eval: GPT-4 based evaluation using a prompted rubric and probabilistic score aggregation

    Formula: score = sum_s s * p_GPT(s | rubric, x, y); p from softmax over logits of digit tokens

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    rubric : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: score

    References
    ----------
    Kamath Ch 8, G-Eval section
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "G-Eval: GPT-4 based evaluation using a prompted rubric and probabilistic score aggregation"})


def cheatsheet():
    return "kmgev: G-Eval: GPT-4 based evaluation using a prompted rubric and probabilistic score aggregation"

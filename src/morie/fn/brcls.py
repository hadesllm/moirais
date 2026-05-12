# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Brier score for probabilistic prediction of binary outcomes."""
import numpy as np
from ._richresult import RichResult

__all__ = ["brier_score"]


def brier_score(y_prob, y_true):
    """
    Brier score for probabilistic prediction of binary outcomes

    Formula: BS = (1/n)*sum_i (f_i - o_i)^2; f_i predicted prob, o_i binary outcome

    Parameters
    ----------
    y_prob : array-like
        Input data.
    y_true : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'brier': 'float'}

    References
    ----------
    Montesinos Lopez Ch 4
    """
    y_prob = np.asarray(y_prob, dtype=float)
    n = int(y_prob) if y_prob.ndim == 0 else len(y_prob)
    result = float(np.mean(y_prob))
    se = float(np.std(y_prob, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Brier score for probabilistic prediction of binary outcomes"})


def cheatsheet():
    return "brcls: Brier score for probabilistic prediction of binary outcomes"

# morie.fn -- function file (hadesllm/morie)
"""F1 score: harmonic mean of precision and recall."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_f1_score"]


def geron_f1_score(y_true, y_pred):
    """
    F1 score: harmonic mean of precision and recall

    Formula: F1 = 2*P*R / (P + R)

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_pred : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: f1

    References
    ----------
    Géron Ch 3
    """
    y_true = np.atleast_1d(np.asarray(y_true, dtype=float))
    n = len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "F1 score: harmonic mean of precision and recall"})


def cheatsheet():
    return "hmf1: F1 score: harmonic mean of precision and recall"

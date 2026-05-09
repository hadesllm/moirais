# moirais.fn — function file (hadesllm/moirais)
"""Soft voting ensemble prediction (argmax mean probability)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_soft_voting"]


def geron_soft_voting(probabilities):
    """
    Soft voting ensemble prediction (argmax mean probability)

    Formula: y_hat = argmax_k (1/L) sum_{l=1..L} p_l(y=k | x)

    Parameters
    ----------
    probabilities : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_pred

    References
    ----------
    Géron Ch 6, Voting Classifier (soft) section
    """
    probabilities = np.asarray(probabilities, dtype=float)
    n = int(probabilities) if probabilities.ndim == 0 else len(probabilities)
    result = float(np.mean(probabilities))
    se = float(np.std(probabilities, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Soft voting ensemble prediction (argmax mean probability)"})


def cheatsheet():
    return "grvots: Soft voting ensemble prediction (argmax mean probability)"

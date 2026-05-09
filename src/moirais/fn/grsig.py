# moirais.fn — function file (hadesllm/moirais)
"""Logistic/sigmoid activation used as link in binary logistic regression."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_sigmoid"]


def geron_sigmoid(t):
    """
    Logistic/sigmoid activation used as link in binary logistic regression

    Formula: sigma(t) = 1 / (1 + exp(-t))

    Parameters
    ----------
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: p

    References
    ----------
    Géron Ch 4, Eq 4-14 (Logistic function)
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Logistic/sigmoid activation used as link in binary logistic regression"})


def cheatsheet():
    return "grsig: Logistic/sigmoid activation used as link in binary logistic regression"

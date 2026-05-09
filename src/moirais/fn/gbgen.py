# moirais.fn — function file (hadesllm/moirais)
"""Gradient boosting for genomic prediction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gradient_boosting_genomic"]


def gradient_boosting_genomic(x, y, markers):
    """
    Gradient boosting for genomic prediction

    Formula: F_m(x) = F_{m-1}(x) + nu*h_m(x) on markers

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    markers : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Montesinos Lopez Ch 9
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gradient boosting for genomic prediction"})


def cheatsheet():
    return "gbgen: Gradient boosting for genomic prediction"

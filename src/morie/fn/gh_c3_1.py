# morie.fn — function file (hadesllm/morie)
"""Random probability measure on Polish space via Kolmogorov consistency theorem."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_random_measure_def"]


def ghosal_random_measure_def(x):
    """
    Random probability measure on Polish space via Kolmogorov consistency theorem

    Formula: G: Omega -> M(X), consistent finite-dimensional distributions via Kolmogorov

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 3 §3.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Random probability measure on Polish space via Kolmogorov consistency theorem"})


def cheatsheet():
    return "gh_c3_1: Random probability measure on Polish space via Kolmogorov consistency theorem"

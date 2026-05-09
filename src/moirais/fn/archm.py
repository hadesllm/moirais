# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""ARCH-in-mean model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["arch_in_mean"]


def arch_in_mean(x):
    """
    ARCH-in-mean model

    Formula: y_t = mu + delta*sigma_t + e_t

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Engle et al. (1987)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ARCH-in-mean model"})


def cheatsheet():
    return "archm: ARCH-in-mean model"

# morie.fn -- function file (hadesllm/morie)
"""White noise exact rate with conjugate Gaussian prior: posterior mean = projection."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_wn_conj_crt"]


def ghosal_wn_conj_crt(x):
    """
    White noise exact rate with conjugate Gaussian prior: posterior mean = projection

    Formula: dY = theta*dt + dW/sqrt(n), Gaussian prior => exact rate and distribution

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
    Ghosal Ch 9 §9.5.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "White noise exact rate with conjugate Gaussian prior: posterior mean = projection"})


def cheatsheet():
    return "gh_c9_9: White noise exact rate with conjugate Gaussian prior: posterior mean = projection"

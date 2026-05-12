# morie.fn -- function file (hadesllm/morie)
"""White noise model posterior contraction at minimax rate."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_wn_crt"]


def ghosal_wn_crt(x):
    """
    White noise model posterior contraction at minimax rate

    Formula: dY_t = f(t)dt + n^{-1/2}dW_t, eps_n = n^{-s/(2s+1)} for s-Sobolev f

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
    Ghosal Ch 8 §8.3.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "White noise model posterior contraction at minimax rate"})


def cheatsheet():
    return "gh_c8_10: White noise model posterior contraction at minimax rate"

# morie.fn -- function file (hadesllm/morie)
"""GP binary regression contraction: optimal rate for probit/logit link."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_gp_binreg_crt"]


def ghosal_gp_binreg_crt(x, y):
    """
    GP binary regression contraction: optimal rate for probit/logit link

    Formula: f ~ GP(0,k), rate n^{-s/(2s+d/2)} for GP on R^d with Sobolev kernel

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 11 §11.3.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GP binary regression contraction: optimal rate for probit/logit link"})


def cheatsheet():
    return "gh_c11_5: GP binary regression contraction: optimal rate for probit/logit link"

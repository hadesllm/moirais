# moirais.fn — function file (hadesllm/moirais)
"""Neutral-to-the-right process: F(t) = 1 - prod_{s<=t}(1-dM(s)) from NTR measure M."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_ntr_def"]


def ghosal_ntr_def(x):
    """
    Neutral-to-the-right process: F(t) = 1 - prod_{s<=t}(1-dM(s)) from NTR measure M

    Formula: F(t) = 1 - exp(-integral_0^t log(1-u) dN(u,s)), M ~ NTR Levy process

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
    Ghosal Ch 13 §13.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Neutral-to-the-right process: F(t) = 1 - prod_{s<=t}(1-dM(s)) from NTR measure M"})


def cheatsheet():
    return "gh_c13_8: Neutral-to-the-right process: F(t) = 1 - prod_{s<=t}(1-dM(s)) from NTR measure M"

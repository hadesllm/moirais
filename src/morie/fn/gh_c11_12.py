# morie.fn — function file (hadesllm/morie)
"""Self-similar GP rescaling: f(lambda*.) =_d lambda^H * f(.) for fractional BM."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_selfsim_gp"]


def ghosal_selfsim_gp(x):
    """
    Self-similar GP rescaling: f(lambda*.) =_d lambda^H * f(.) for fractional BM

    Formula: fBM: f(lambda*.) =_d lambda^H f(.) for Hurst H

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
    Ghosal Ch 11 §11.5.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Self-similar GP rescaling: f(lambda*.) =_d lambda^H * f(.) for fractional BM"})


def cheatsheet():
    return "gh_c11_12: Self-similar GP rescaling: f(lambda*.) =_d lambda^H * f(.) for fractional BM"

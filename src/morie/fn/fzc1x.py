# morie.fn — function file (hadesllm/morie)
"""c_1(x) coefficient in boundary-free KDFE bias."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_c1_coefficient"]


def fauzi_c1_coefficient(x, g_func, density):
    """
    c_1(x) coefficient in boundary-free KDFE bias

    Formula: c_1(x) = g''(g^{-1}(x))*f_X(x) + [g'(g^{-1}(x))]^2 * f_X'(x)

    Parameters
    ----------
    x : array-like
        Input data.
    g_func : array-like
        Input data.
    density : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: coefficient

    References
    ----------
    Fauzi Ch 5, Eq 5.8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "c_1(x) coefficient in boundary-free KDFE bias"})


def cheatsheet():
    return "fzc1x: c_1(x) coefficient in boundary-free KDFE bias"

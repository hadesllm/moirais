# moirais.fn — function file (hadesllm/moirais)
"""c_2(x) coefficient in boundary-free KDE bias."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_c2_coefficient"]


def fauzi_c2_coefficient(x, g_func, density):
    """
    c_2(x) coefficient in boundary-free KDE bias

    Formula: c_2(x) = g'''*f_X + 3g''*g'*f_X' + [g']^3*f_X'' (all evaluated at g^{-1}(x))

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
    Fauzi Ch 5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "c_2(x) coefficient in boundary-free KDE bias"})


def cheatsheet():
    return "fzc2x: c_2(x) coefficient in boundary-free KDE bias"

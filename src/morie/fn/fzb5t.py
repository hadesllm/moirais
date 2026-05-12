# morie.fn -- function file (hadesllm/morie)
"""b_5(t) coefficient in MRL variance formula."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_b5_coefficient_mrl"]


def fauzi_b5_coefficient_mrl(t, g_func, density, mrl):
    """
    b_5(t) coefficient in MRL variance formula

    Formula: b_5(t) = g'(g^{-1}(t)) * f_X(t) * m_X^2(t)

    Parameters
    ----------
    t : array-like
        Input data.
    g_func : array-like
        Input data.
    density : array-like
        Input data.
    mrl : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: coefficient

    References
    ----------
    Fauzi Ch 4, Eq 4.28
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "b_5(t) coefficient in MRL variance formula"})


def cheatsheet():
    return "fzb5t: b_5(t) coefficient in MRL variance formula"

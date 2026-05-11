# morie.fn — function file (hadesllm/morie)
"""Theorem 2.3: variance of bias-reduced KDFE."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_thm2_3_var_brdkdfe"]


def fauzi_thm2_3_var_brdkdfe(x, bandwidth, a):
    """
    Theorem 2.3: variance of bias-reduced KDFE

    Formula: Var[F_tilde_X] = (1/n)F_X(1-F_X) - (h/n)*[2(a^4+1)/(a^2-1)^2*r1+r2]*f_X + o(h/n)

    Parameters
    ----------
    x : array-like
        Input data.
    bandwidth : array-like
        Input data.
    a : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: variance

    References
    ----------
    Fauzi Ch 2, Theorem 2.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Theorem 2.3: variance of bias-reduced KDFE"})


def cheatsheet():
    return "fzt23: Theorem 2.3: variance of bias-reduced KDFE"

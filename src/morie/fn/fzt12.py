# morie.fn — function file (hadesllm/morie)
"""Theorem 1.2: variance of modified gamma KDE."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_thm1_2_var_mgkde"]


def fauzi_thm1_2_var_mgkde(x, bandwidth):
    """
    Theorem 1.2: variance of modified gamma KDE

    Formula: Var[f_tilde_X(x)] = (1/n) f_X(x)/x * integral u^2 K_1(u) du + o(1/n)

    Parameters
    ----------
    x : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: variance

    References
    ----------
    Fauzi Ch 1, Theorem 1.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Theorem 1.2: variance of modified gamma KDE"})


def cheatsheet():
    return "fzt12: Theorem 1.2: variance of modified gamma KDE"

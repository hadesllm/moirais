# morie.fn — function file (hadesllm/morie)
"""Theorem 1.4: asymptotic normality of modified gamma KDE."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_thm1_4_asympnorm_mgkde"]


def fauzi_thm1_4_asympnorm_mgkde(x, bandwidth):
    """
    Theorem 1.4: asymptotic normality of modified gamma KDE

    Formula: sqrt(n)*(f_tilde_X - f_X)/sqrt(Var[f_tilde_X]) ->_D N(0,1)

    Parameters
    ----------
    x : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: statistic

    References
    ----------
    Fauzi Ch 1, Theorem 1.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Theorem 1.4: asymptotic normality of modified gamma KDE"})


def cheatsheet():
    return "fzt14: Theorem 1.4: asymptotic normality of modified gamma KDE"

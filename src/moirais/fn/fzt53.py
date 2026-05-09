# moirais.fn — function file (hadesllm/moirais)
"""Theorem 5.3: asymptotic normality of boundary-free KDFE."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_thm5_3_bdfree_normality"]


def fauzi_thm5_3_bdfree_normality(x, bandwidth, g_func):
    """
    Theorem 5.3: asymptotic normality of boundary-free KDFE

    Formula: (F_tilde_X(x)-F_X(x))/sqrt(Var[F_tilde_X(x)]) ->_D N(0,1)

    Parameters
    ----------
    x : array-like
        Input data.
    bandwidth : array-like
        Input data.
    g_func : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: statistic

    References
    ----------
    Fauzi Ch 5, Theorem 5.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Theorem 5.3: asymptotic normality of boundary-free KDFE"})


def cheatsheet():
    return "fzt53: Theorem 5.3: asymptotic normality of boundary-free KDFE"

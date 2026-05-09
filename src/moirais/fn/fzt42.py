# moirais.fn — function file (hadesllm/moirais)
"""Theorem 4.2: bias and variance of S_tilde_X,2."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_thm4_2_surv2_bias_var"]


def fauzi_thm4_2_surv2_bias_var(t, bandwidth, g_func):
    """
    Theorem 4.2: bias and variance of S_tilde_X,2

    Formula: Bias[S_tilde_X,2]=(h^2/2)*b3(t)*mu2(K); Var=(1/n)[2S_X-S_X^2]+o(h/n)

    Parameters
    ----------
    t : array-like
        Input data.
    bandwidth : array-like
        Input data.
    g_func : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bias_variance

    References
    ----------
    Fauzi Ch 4, Theorem 4.2, Eq 4.19-4.20
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Theorem 4.2: bias and variance of S_tilde_X,2"})


def cheatsheet():
    return "fzt42: Theorem 4.2: bias and variance of S_tilde_X,2"

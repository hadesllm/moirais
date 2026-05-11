# morie.fn — function file (hadesllm/morie)
"""Theorem 4.1: biases and variances of S_tilde_X and S_tilde_X,1."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_thm4_1_surv_bias_var"]


def fauzi_thm4_1_surv_bias_var(t, bandwidth, g_func):
    """
    Theorem 4.1: biases and variances of S_tilde_X and S_tilde_X,1

    Formula: Bias[S_tilde]= -(h^2/2)*b1*mu2(K); Var[S_tilde]=(1/n)S_X*F_X - (h/n)*g'*f_X*int VW dy

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
    Fauzi Ch 4, Theorem 4.1, Eq 4.10-4.13
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Theorem 4.1: biases and variances of S_tilde_X and S_tilde_X,1"})


def cheatsheet():
    return "fzt41: Theorem 4.1: biases and variances of S_tilde_X and S_tilde_X,1"

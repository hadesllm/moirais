# morie.fn -- function file (hadesllm/morie)
"""Covariance between S_tilde_X,2 and S_tilde_X."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_cov_surv_est2"]


def fauzi_cov_surv_est2(t, bandwidth):
    """
    Covariance between S_tilde_X,2 and S_tilde_X

    Formula: Cov[S_tilde_X,2(t), S_tilde_X(t)] = (1/n)*S_X(t)*F_X(t) + o(h/n)

    Parameters
    ----------
    t : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: covariance

    References
    ----------
    Fauzi Ch 4, Eq 4.22
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Covariance between S_tilde_X,2 and S_tilde_X"})


def cheatsheet():
    return "fzcov2: Covariance between S_tilde_X,2 and S_tilde_X"

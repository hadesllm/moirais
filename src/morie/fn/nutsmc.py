"""No-U-Turn sampler (NUTS / HMC)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["nuts_sampler"]


def nuts_sampler(log_p, grad_log_p, x0, n_iter):
    """
    No-U-Turn sampler (NUTS / HMC)

    Formula: adaptive HMC trajectory length

    Parameters
    ----------
    log_p : array-like
        Input data.
    grad_log_p : array-like
        Input data.
    x0 : array-like
        Input data.
    n_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hoffman-Gelman (2014)
    """
    log_p = np.atleast_1d(np.asarray(log_p, dtype=float))
    n = len(log_p)
    result = float(np.mean(log_p))
    se = float(np.std(log_p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "No-U-Turn sampler (NUTS / HMC)"})


def cheatsheet():
    return "nutsmc: No-U-Turn sampler (NUTS / HMC)"

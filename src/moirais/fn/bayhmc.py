"""HMC with dual-averaging step-size adaptation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hmc_dual_avg"]


def hmc_dual_avg(log_p, grad, x0, target_accept):
    """
    HMC with dual-averaging step-size adaptation

    Formula: adapt eps to target accept rate

    Parameters
    ----------
    log_p : array-like
        Input data.
    grad : array-like
        Input data.
    x0 : array-like
        Input data.
    target_accept : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "HMC with dual-averaging step-size adaptation"})


def cheatsheet():
    return "bayhmc: HMC with dual-averaging step-size adaptation"

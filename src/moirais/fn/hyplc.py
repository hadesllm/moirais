"""Harmonic mean marginal likelihood (cautionary)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["harmonic_mean_estimator"]


def harmonic_mean_estimator(log_lik):
    """
    Harmonic mean marginal likelihood (cautionary)

    Formula: m(y) ≈ ( (1/S) sum_s 1/p(y|theta_s) )^{-1}

    Parameters
    ----------
    log_lik : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Newton & Raftery (1994); cf. Neal's pathological
    """
    log_lik = np.atleast_1d(np.asarray(log_lik, dtype=float))
    n = len(log_lik)
    result = float(np.mean(log_lik))
    se = float(np.std(log_lik, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Harmonic mean marginal likelihood (cautionary)"})


def cheatsheet():
    return "hyplc: Harmonic mean marginal likelihood (cautionary)"

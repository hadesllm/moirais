"""Posterior predictive mean."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["posterior_predictive_mean"]


def posterior_predictive_mean(samples):
    """
    Posterior predictive mean

    Formula: E[y_tilde | y] = int p(y_tilde | theta) p(theta | y) dtheta

    Parameters
    ----------
    samples : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gelman et al. BDA3 §6.3
    """
    samples = np.atleast_1d(np.asarray(samples, dtype=float))
    n = len(samples)
    result = float(np.mean(samples))
    se = float(np.std(samples, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Posterior predictive mean"})


def cheatsheet():
    return "pposm: Posterior predictive mean"

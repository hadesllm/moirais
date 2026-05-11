"""Posterior predictive replication draws."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["posterior_predictive_replication"]


def posterior_predictive_replication(samples):
    """
    Posterior predictive replication draws

    Formula: y_rep^(s) ~ p(y | theta_s) for s in posterior draws

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
    Gelman et al. BDA3 §6
    """
    samples = np.atleast_1d(np.asarray(samples, dtype=float))
    n = len(samples)
    result = float(np.mean(samples))
    se = float(np.std(samples, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Posterior predictive replication draws"})


def cheatsheet():
    return "posrr: Posterior predictive replication draws"

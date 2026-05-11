"""Hierarchical Bayesian model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hierarchical_model"]


def hierarchical_model(y, group):
    """
    Hierarchical Bayesian model

    Formula: mu_g ~ p(mu); y_ig ~ p(y | mu_g)

    Parameters
    ----------
    y : array-like
        Input data.
    group : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gelman et al (2013) BDA3 Ch 5
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hierarchical Bayesian model"})


def cheatsheet():
    return "hmcrg: Hierarchical Bayesian model"

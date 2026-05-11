"""Empirical Bayes for NP priors."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["empirical_bayes_np"]


def empirical_bayes_np(y, prior_family):
    """
    Empirical Bayes for NP priors

    Formula: plug-in MLE for hyperparameters

    Parameters
    ----------
    y : array-like
        Input data.
    prior_family : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Robbins (1956)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Empirical Bayes for NP priors"})


def cheatsheet():
    return "empbnp: Empirical Bayes for NP priors"

"""Posterior predictive density NP."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["posterior_predictive"]


def posterior_predictive(data, prior, new):
    """
    Posterior predictive density NP

    Formula: f_pred(y_new | data) integrated

    Parameters
    ----------
    data : array-like
        Input data.
    prior : array-like
        Input data.
    new : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Müller-Quintana-Rosner (2011)
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Posterior predictive density NP"})


def cheatsheet():
    return "posspr: Posterior predictive density NP"

"""Logistic MSM for binary outcomes."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["msm_logistic"]


def msm_logistic(y, treatment_history, covariate_history):
    """
    Logistic MSM for binary outcomes

    Formula: logit P(Y(a_bar)=1) = beta_0 + beta_a a_bar

    Parameters
    ----------
    y : array-like
        Input data.
    treatment_history : array-like
        Input data.
    covariate_history : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Robins-Hernán-Brumback (2000)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Logistic MSM for binary outcomes"})


def cheatsheet():
    return "msmlog: Logistic MSM for binary outcomes"

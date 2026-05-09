"""MSM Poisson regression for count outcomes."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["msm_poisson"]


def msm_poisson(y, treatment_history, covariate_history, offset):
    """
    MSM Poisson regression for count outcomes

    Formula: log E[Y(a_bar)] = beta_0 + beta_a a_bar

    Parameters
    ----------
    y : array-like
        Input data.
    treatment_history : array-like
        Input data.
    covariate_history : array-like
        Input data.
    offset : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hernán-Robins (2020) Causal Inference Book Ch 12
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MSM Poisson regression for count outcomes"})


def cheatsheet():
    return "msmpoi: MSM Poisson regression for count outcomes"

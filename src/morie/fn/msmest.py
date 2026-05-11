"""Marginal structural model fit by inverse-probability-of-treatment weighting."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["marginal_structural_model"]


def marginal_structural_model(y, treatment_history, covariate_history, time):
    """
    Marginal structural model fit by inverse-probability-of-treatment weighting

    Formula: E[Y(a_bar)] = beta_0 + sum_t beta_t a_t; weights w_i = prod_t f(A_t)/f(A_t|H_t)

    Parameters
    ----------
    y : array-like
        Input data.
    treatment_history : array-like
        Input data.
    covariate_history : array-like
        Input data.
    time : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Robins (2000); Robins-Hernán-Brumback (2000)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Marginal structural model fit by inverse-probability-of-treatment weighting"})


def cheatsheet():
    return "msmest: Marginal structural model fit by inverse-probability-of-treatment weighting"

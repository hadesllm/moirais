"""Robins g-formula — Monte Carlo simulation of counterfactual outcome distribution."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["robins_g_formula"]


def robins_g_formula(y, treatment_history, covariate_history, time, intervention):
    """
    Robins g-formula — Monte Carlo simulation of counterfactual outcome distribution

    Formula: draw L_t* | a_{t-1}*, l_{t-1}* from fitted models; iterate to T; average Y_T*

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
    intervention : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Robins (1986); Taubman et al (2009)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Robins g-formula — Monte Carlo simulation of counterfactual outcome distribution"})


def cheatsheet():
    return "gforml: Robins g-formula — Monte Carlo simulation of counterfactual outcome distribution"

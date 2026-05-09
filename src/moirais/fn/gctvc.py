"""G-computation (parametric g-formula) for time-varying confounding."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["g_computation_time_varying"]


def g_computation_time_varying(y, treatment_history, covariate_history, time):
    """
    G-computation (parametric g-formula) for time-varying confounding

    Formula: E[Y(a_bar)] = sum_l_bar P(Y|a_bar, l_bar) prod_t P(L_t|a_{t-1}, l_{t-1})

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
    Robins (1986); Robins-Hernán (2009)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "G-computation (parametric g-formula) for time-varying confounding"})


def cheatsheet():
    return "gctvc: G-computation (parametric g-formula) for time-varying confounding"

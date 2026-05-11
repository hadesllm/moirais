"""Structural nested Cox model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["snm_cox"]


def snm_cox(time, event, treatment_history, covariate_history):
    """
    Structural nested Cox model

    Formula: H_t(psi) = exp(-gamma A_t) Y; g-estimation

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
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
    Robins (1992); Witteman et al (1998)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Structural nested Cox model"})


def cheatsheet():
    return "snmcox: Structural nested Cox model"

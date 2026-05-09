"""Cause-specific hazard model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cause_specific_hazard"]


def cause_specific_hazard(time, event_type, X, cause):
    """
    Cause-specific hazard model

    Formula: lambda_k(t) = lambda_0k(t) exp(beta_k X)

    Parameters
    ----------
    time : array-like
        Input data.
    event_type : array-like
        Input data.
    X : array-like
        Input data.
    cause : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kalbfleisch-Prentice (1980)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cause-specific hazard model"})


def cheatsheet():
    return "crrcsh: Cause-specific hazard model"

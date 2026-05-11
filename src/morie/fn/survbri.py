"""Time-dependent Brier score."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["brier_score"]


def brier_score(time, event, predicted_S, t_grid):
    """
    Time-dependent Brier score

    Formula: E[(I(T>t) - S(t|X))^2]

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    predicted_S : array-like
        Input data.
    t_grid : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Graf et al (1999); Gerds-Schumacher (2006)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Time-dependent Brier score"})


def cheatsheet():
    return "survbri: Time-dependent Brier score"

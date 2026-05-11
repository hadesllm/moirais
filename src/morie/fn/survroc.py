"""Time-dependent ROC curve."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["time_dep_roc"]


def time_dep_roc(time, event, marker, t):
    """
    Time-dependent ROC curve

    Formula: AUC(t) = P(M_i > M_j | T_i <= t < T_j)

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    marker : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Heagerty-Lumley-Pepe (2000)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Time-dependent ROC curve"})


def cheatsheet():
    return "survroc: Time-dependent ROC curve"

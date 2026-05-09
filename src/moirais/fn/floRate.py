"""Flow-duration curve."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["flow_duration"]


def flow_duration(Q):
    """
    Flow-duration curve

    Formula: sorted Q_t vs exceedance probability

    Parameters
    ----------
    Q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Vogel-Fennessey (1995)
    """
    Q = np.atleast_1d(np.asarray(Q, dtype=float))
    n = len(Q)
    result = float(np.mean(Q))
    se = float(np.std(Q, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Flow-duration curve"})


def cheatsheet():
    return "floRate: Flow-duration curve"

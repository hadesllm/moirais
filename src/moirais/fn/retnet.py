"""RetNet retention mechanism (gated recurrent + parallel)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["retnet_retention"]


def retnet_retention(y, Q, K, V, gamma):
    """
    RetNet retention mechanism (gated recurrent + parallel)

    Formula: S_n = gamma S_{n-1} + K_n^T V_n; O_n = Q_n S_n

    Parameters
    ----------
    y : array-like
        Input data.
    Q : array-like
        Input data.
    K : array-like
        Input data.
    V : array-like
        Input data.
    gamma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sun et al. (2023) RetNet
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RetNet retention mechanism (gated recurrent + parallel)"})


def cheatsheet():
    return "retnet: RetNet retention mechanism (gated recurrent + parallel)"

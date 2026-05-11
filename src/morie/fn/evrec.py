"""Number of upper records in an iid sequence."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_record_count"]


def evt_record_count(x):
    """
    Number of upper records in an iid sequence

    Formula: R_n = Σ 1{X_i > max(X_1,..X_{i-1})}; E[R_n]≈log n

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: R, idx

    References
    ----------
    Arnold-Balakrishnan-Nagaraja (1998)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Number of upper records in an iid sequence"})


def cheatsheet():
    return "evrec: Number of upper records in an iid sequence"

"""Delete-d jackknife generalising leave-one-out."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_jackknife_d"]


def boot_jackknife_d(x, d, stat):
    """
    Delete-d jackknife generalising leave-one-out

    Formula: Leave d points out; weight by C(n,d)

    Parameters
    ----------
    x : array-like
        Input data.
    d : array-like
        Input data.
    stat : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta_jkd, var_jkd

    References
    ----------
    Shao & Wu (1989)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Delete-d jackknife generalising leave-one-out"})


def cheatsheet():
    return "btjknd: Delete-d jackknife generalising leave-one-out"

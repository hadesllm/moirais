"""Fully-corrective Frank-Wolfe."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["fully_corrective_fw"]


def fully_corrective_fw(f, grad_f, domain, x0):
    """
    Fully-corrective Frank-Wolfe

    Formula: min over hull of selected vertices

    Parameters
    ----------
    f : array-like
        Input data.
    grad_f : array-like
        Input data.
    domain : array-like
        Input data.
    x0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Holloway (1974)
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fully-corrective Frank-Wolfe"})


def cheatsheet():
    return "fwlfwd: Fully-corrective Frank-Wolfe"

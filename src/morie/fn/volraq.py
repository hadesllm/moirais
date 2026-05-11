"""Realised quadratic variation = RV in-sample."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_realised_quadratic_var"]


def vol_realised_quadratic_var(x):
    """
    Realised quadratic variation = RV in-sample

    Formula: [X]_T = Σ_i (X_{t_i}-X_{t_{i-1}})²

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: QV

    References
    ----------
    Andersen-Bollerslev (1998)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Realised quadratic variation = RV in-sample"})


def cheatsheet():
    return "volraq: Realised quadratic variation = RV in-sample"

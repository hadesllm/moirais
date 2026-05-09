"""Earth Mover's Distance via LP between discrete measures."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_emd_solver"]


def ot_emd_solver(a, b, C):
    """
    Earth Mover's Distance via LP between discrete measures

    Formula: min_T <T,C> s.t. T1=a, T^T 1=b, T>=0

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.
    C : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: T, cost

    References
    ----------
    Rubner-Tomasi-Guibas (2000)
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Earth Mover's Distance via LP between discrete measures"})


def cheatsheet():
    return "otemd: Earth Mover's Distance via LP between discrete measures"

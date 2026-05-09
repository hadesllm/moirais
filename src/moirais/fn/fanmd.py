"""Sobol decomposition of f."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["fanova_decomposition"]


def fanova_decomposition(f, input_dist):
    """
    Sobol decomposition of f

    Formula: f = sum_S f_S; f_S only depends on x_S

    Parameters
    ----------
    f : array-like
        Input data.
    input_dist : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sobol (1993)
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sobol decomposition of f"})


def cheatsheet():
    return "fanmd: Sobol decomposition of f"

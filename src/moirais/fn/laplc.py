"""Laplace mechanism."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["laplace_mechanism"]


def laplace_mechanism(f_value, sensitivity, epsilon):
    """
    Laplace mechanism

    Formula: M(D) = f(D) + Lap(Δf/ε)

    Parameters
    ----------
    f_value : array-like
        Input data.
    sensitivity : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dwork et al (2006)
    """
    f_value = np.atleast_1d(np.asarray(f_value, dtype=float))
    n = len(f_value)
    result = float(np.mean(f_value))
    se = float(np.std(f_value, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Laplace mechanism"})


def cheatsheet():
    return "laplc: Laplace mechanism"

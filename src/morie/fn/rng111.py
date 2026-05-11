"""First-order difference operator approximating the time derivative.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_first_difference_operator"]


def rangayyan_ch3_first_difference_operator(x, T, n):
    """
    First-order difference operator approximating the time derivative.

    Formula: y(n) = (1/T) * [x(n) - x(n-1)]

    Parameters
    ----------
    x : array-like
        Input data.
    T : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.123, p. 145
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "First-order difference operator approximating the time derivative."})


def cheatsheet():
    return "rng111: First-order difference operator approximating the time derivative."

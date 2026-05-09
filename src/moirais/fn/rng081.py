"""Even-symmetric part of a signal.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_even_part"]


def rangayyan_ch3_even_part(x, n):
    """
    Even-symmetric part of a signal.

    Formula: x_e(n) = 0.5 * [x(n) + x(-n)]

    Parameters
    ----------
    x : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.92, p. 135
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Even-symmetric part of a signal."})


def cheatsheet():
    return "rng081: Even-symmetric part of a signal."

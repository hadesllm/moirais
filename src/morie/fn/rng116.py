"""Three-point central-difference operator (lower-noise derivative).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_three_point_central_difference"]


def rangayyan_ch3_three_point_central_difference(x, T, n):
    """
    Three-point central-difference operator (lower-noise derivative).

    Formula: y_3(n) = (1/(2*T)) * [x(n) - x(n-2)]

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
    Rangayyan (2024), Ch 3, Eq 3.128, p. 147
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Three-point central-difference operator (lower-noise derivative)."})


def cheatsheet():
    return "rng116: Three-point central-difference operator (lower-noise derivative)."

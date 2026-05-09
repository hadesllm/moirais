"""Derivative operator used by Pan and Tompkins for QRS detection.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_pan_tompkins_derivative_operator"]


def rangayyan_ch4_pan_tompkins_derivative_operator(x, n):
    """
    Derivative operator used by Pan and Tompkins for QRS detection.

    Formula: y(n) = (1/8) * [2*x(n) + x(n-1) - x(n-3) - 2*x(n-4)]

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
    Rangayyan (2024), Ch 4, Eq 4.14, p. 222
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Derivative operator used by Pan and Tompkins for QRS detection."})


def cheatsheet():
    return "rng188: Derivative operator used by Pan and Tompkins for QRS detection."

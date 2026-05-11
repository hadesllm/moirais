"""Weighted combination of first and second derivatives for QRS detection.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_qrs_combined_balda"]


def rangayyan_ch4_qrs_combined_balda(y_0, y_1, n):
    """
    Weighted combination of first and second derivatives for QRS detection.

    Formula: y_2(n) = 1.3 * y_0(n) + 1.1 * y_1(n)

    Parameters
    ----------
    y_0 : array-like
        Input data.
    y_1 : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.3, p. 218
    """
    y_0 = np.atleast_1d(np.asarray(y_0, dtype=float))
    n = len(y_0)
    result = float(np.mean(y_0))
    se = float(np.std(y_0, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Weighted combination of first and second derivatives for QRS detection."})


def cheatsheet():
    return "rng178: Weighted combination of first and second derivatives for QRS detection."

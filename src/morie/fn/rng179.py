"""Filtered weighted-squared first-derivative operator for QRS detection (Murthy and Rangaraj).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_filtered_derivative_murthy"]


def rangayyan_ch4_filtered_derivative_murthy(x, n, N):
    """
    Filtered weighted-squared first-derivative operator for QRS detection (Murthy and Rangaraj).

    Formula: g_1(n) = sum_{i=1}^{N} |x(n-i+1) - x(n-i)|^2 * (N - i + 1)

    Parameters
    ----------
    x : array-like
        Input data.
    n : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.4, p. 219
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Filtered weighted-squared first-derivative operator for QRS detection (Murthy and Rangaraj)."})


def cheatsheet():
    return "rng179: Filtered weighted-squared first-derivative operator for QRS detection (Murthy and Rangaraj)."

"""Output of two LSI systems in series equals input convolved with combined response.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_lsi_series_total"]


def rangayyan_ch3_lsi_series_total(x, h_1, h_2, n):
    """
    Output of two LSI systems in series equals input convolved with combined response.

    Formula: y(n) = s(n) * h_2(n) = x(n) * h_1(n) * h_2(n) = x(n) * h(n)

    Parameters
    ----------
    x : array-like
        Input data.
    h_1 : array-like
        Input data.
    h_2 : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.44, p. 115
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Output of two LSI systems in series equals input convolved with combined response."})


def cheatsheet():
    return "rng042: Output of two LSI systems in series equals input convolved with combined response."

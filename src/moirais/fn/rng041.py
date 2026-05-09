"""Intermediate output of the first LSI system in a series cascade.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_lsi_series_intermediate"]


def rangayyan_ch3_lsi_series_intermediate(x, h_1, n):
    """
    Intermediate output of the first LSI system in a series cascade.

    Formula: s(n) = x(n) * h_1(n)

    Parameters
    ----------
    x : array-like
        Input data.
    h_1 : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.43, p. 115
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Intermediate output of the first LSI system in a series cascade."})


def cheatsheet():
    return "rng041: Intermediate output of the first LSI system in a series cascade."

r"""Demographic representation.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch6_demographic_representation"]


def kamath_ch6_demographic_representation(G_i, A_i, Yhat):
    r"""
    Demographic representation.

    Formula: \mathrm{DR}(G_i) = \sum_{a_i\in A_i}\sum_{\hat{Y}\in\hat{Y}} C(a_i,\hat{Y})

    Parameters
    ----------
    G_i : array-like
        Input data.
    A_i : array-like
        Input data.
    Yhat : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 6, Eq 6.15, p. 236
    r"""
    G_i = np.atleast_1d(np.asarray(G_i, dtype=float))
    n = len(G_i)
    result = float(np.mean(G_i))
    se = float(np.std(G_i, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Demographic representation."})


def cheatsheet():
    return "km091: Demographic representation."

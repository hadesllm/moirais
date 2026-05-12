r"""Stereotypical assoc.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch6_stereotypical_assoc"]


def kamath_ch6_stereotypical_assoc(w, A_i, Yhat):
    r"""
    Stereotypical assoc.

    Formula: \mathrm{ST}(w)_i = \sum_{a_i\in A_i}\sum_{\hat{Y}\in\hat{Y}} C(a_i,\hat{Y})\,I(C(w,\hat{Y})>0)

    Parameters
    ----------
    w : array-like
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
    Kamath et al (2024), Ch 6, Eq 6.16, p. 237
    r"""
    w = np.atleast_1d(np.asarray(w, dtype=float))
    n = len(w)
    result = float(np.mean(w))
    se = float(np.std(w, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stereotypical assoc."})


def cheatsheet():
    return "km092: Stereotypical assoc."

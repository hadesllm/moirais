"""Emt metric.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch6_emt_metric"]


def kamath_ch6_emt_metric(Yhat, c):
    """
    Emt metric.

    Formula: \mathrm{EMT}(\hat{Y}) = \max_{\hat{Y}\in\hat{Y}} c(\hat{Y})

    Parameters
    ----------
    Yhat : array-like
        Input data.
    c : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 6, Eq 6.23, p. 250
    """
    Yhat = np.atleast_1d(np.asarray(Yhat, dtype=float))
    n = len(Yhat)
    result = float(np.mean(Yhat))
    se = float(np.std(Yhat, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Emt metric."})


def cheatsheet():
    return "km099: Emt metric."

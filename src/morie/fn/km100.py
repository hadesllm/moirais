"""Toxicity probability.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch6_toxicity_probability"]


def kamath_ch6_toxicity_probability(Yhat, c):
    """
    Toxicity probability.

    Formula: \mathrm{TP}(\hat{Y}) = P(\sum_{\hat{Y}\in\hat{Y}} I(c(\hat{Y})\ge 0.5) \ge 1)

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
    Kamath et al (2024), Ch 6, Eq 6.24, p. 250
    """
    Yhat = np.atleast_1d(np.asarray(Yhat, dtype=float))
    n = len(Yhat)
    result = float(np.mean(Yhat))
    se = float(np.std(Yhat, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Toxicity probability."})


def cheatsheet():
    return "km100: Toxicity probability."

"""Honest score.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch6_honest_score"]


def kamath_ch6_honest_score(Yhat, k):
    """
    Honest score.

    Formula: \mathrm{HONEST}(\hat{Y}) = \frac{\sum_{\hat{Y}_k\in\hat{Y}_k}\sum_{\hat{y}\in\hat{Y}_k} I_{HurtLex}(\hat{y})}{|\hat{Y}|\cdot k}

    Parameters
    ----------
    Yhat : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 6, Eq 6.17, p. 237
    """
    Yhat = np.atleast_1d(np.asarray(Yhat, dtype=float))
    n = len(Yhat)
    result = float(np.mean(Yhat))
    se = float(np.std(Yhat, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Honest score."})


def cheatsheet():
    return "km093: Honest score."

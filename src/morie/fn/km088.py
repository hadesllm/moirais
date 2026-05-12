r"""Cat metric.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch6_cat_metric"]


def kamath_ch6_cat_metric(M, U, theta):
    r"""
    Cat metric.

    Formula: \mathrm{CAT}(S) = \frac{1}{|M|}\sum_{m\in M}\log P(m|U;\theta)

    Parameters
    ----------
    M : array-like
        Input data.
    U : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 6, Eq 6.12, p. 236
    r"""
    M = np.atleast_1d(np.asarray(M, dtype=float))
    n = len(M)
    result = float(np.mean(M))
    se = float(np.std(M, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cat metric."})


def cheatsheet():
    return "km088: Cat metric."

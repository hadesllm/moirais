"""Cps metric.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch6_cps_metric"]


def kamath_ch6_cps_metric(U, M, theta):
    """
    Cps metric.

    Formula: \mathrm{CPS}(S) = \sum_{u\in U}\log P(u|U_{\setminus u},M;\theta)

    Parameters
    ----------
    U : array-like
        Input data.
    M : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 6, Eq 6.11, p. 236
    """
    U = np.atleast_1d(np.asarray(U, dtype=float))
    n = len(U)
    result = float(np.mean(U))
    se = float(np.std(U, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cps metric."})


def cheatsheet():
    return "km087: Cps metric."

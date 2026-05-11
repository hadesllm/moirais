"""Cbs variance.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch6_cbs_variance"]


def kamath_ch6_cbs_variance(W, A, p_a, p_prior):
    """
    Cbs variance.

    Formula: \mathrm{CBS}(S) = \frac{1}{|W|}\sum_{w\in W} \mathrm{Var}_{a\in A}\log\frac{p_a}{p_{prior}}

    Parameters
    ----------
    W : array-like
        Input data.
    A : array-like
        Input data.
    p_a : array-like
        Input data.
    p_prior : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 6, Eq 6.9, p. 235
    """
    W = np.atleast_1d(np.asarray(W, dtype=float))
    n = len(W)
    result = float(np.mean(W))
    se = float(np.std(W, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cbs variance."})


def cheatsheet():
    return "km085: Cbs variance."

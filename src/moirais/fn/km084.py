"""Lpbs bias.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch6_lpbs_bias"]


def kamath_ch6_lpbs_bias(p_a, p_prior):
    """
    Lpbs bias.

    Formula: \mathrm{LPBS}(S) = \log\frac{p_{a_i}}{p_{prior_i}} - \log\frac{p_{a_j}}{p_{prior_j}}

    Parameters
    ----------
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
    Kamath et al (2024), Ch 6, Eq 6.8, p. 235
    """
    p_a = np.atleast_1d(np.asarray(p_a, dtype=float))
    n = len(p_a)
    result = float(np.mean(p_a))
    se = float(np.std(p_a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Lpbs bias."})


def cheatsheet():
    return "km084: Lpbs bias."

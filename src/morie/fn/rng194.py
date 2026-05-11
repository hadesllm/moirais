"""Heart rate computed from average RR interval (in seconds).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_heart_rate_from_rr"]


def rangayyan_ch4_heart_rate_from_rr(RR_a):
    """
    Heart rate computed from average RR interval (in seconds).

    Formula: HR = 60 / RR_a

    Parameters
    ----------
    RR_a : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.20, p. 224
    """
    RR_a = np.atleast_1d(np.asarray(RR_a, dtype=float))
    n = len(RR_a)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Heart rate computed from average RR interval (in seconds)."})
    estimate = np.median(RR_a)
    se = 1.2533 * np.std(RR_a, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Heart rate computed from average RR interval (in seconds)."})


def cheatsheet():
    return "rng194: Heart rate computed from average RR interval (in seconds)."

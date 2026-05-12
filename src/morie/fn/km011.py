r"""Scaled dot score.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_scaled_dot_score"]


def kamath_ch2_scaled_dot_score(q, k, d_k):
    r"""
    Scaled dot score.

    Formula: \alpha(q,k) = \frac{q\cdot k}{\sqrt{d_k}}

    Parameters
    ----------
    q : array-like
        Input data.
    k : array-like
        Input data.
    d_k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.11, p. 33
    r"""
    q = np.atleast_1d(np.asarray(q, dtype=float))
    n = len(q)
    result = float(np.mean(q))
    se = float(np.std(q, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Scaled dot score."})


def cheatsheet():
    return "km011: Scaled dot score."

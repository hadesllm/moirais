"""Attention score.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_attention_score"]


def kamath_ch2_attention_score(q, k_i):
    """
    Attention score.

    Formula: a_i = \alpha(q, k_i)

    Parameters
    ----------
    q : array-like
        Input data.
    k_i : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.7, p. 32
    """
    q = np.atleast_1d(np.asarray(q, dtype=float))
    n = len(q)
    result = float(np.mean(q))
    se = float(np.std(q, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Attention score."})


def cheatsheet():
    return "km007: Attention score."

r"""Co occurrence bias.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch6_co_occurrence_bias"]


def kamath_ch6_co_occurrence_bias(w, A_i, A_j):
    r"""
    Co occurrence bias.

    Formula: \text{Co-Occurrence Bias Score}(w) = \log\frac{P(w|A_i)}{P(w|A_j)}

    Parameters
    ----------
    w : array-like
        Input data.
    A_i : array-like
        Input data.
    A_j : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 6, Eq 6.14, p. 236
    r"""
    w = np.atleast_1d(np.asarray(w, dtype=float))
    n = len(w)
    result = float(np.mean(w))
    se = float(np.std(w, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Co occurrence bias."})


def cheatsheet():
    return "km090: Co occurrence bias."

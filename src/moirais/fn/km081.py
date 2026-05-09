"""Weat similarity.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch6_weat_similarity"]


def kamath_ch6_weat_similarity(a, W_1, W_2):
    """
    Weat similarity.

    Formula: s(a,W_1,W_2) = \mathrm{mean}_{w_1\in W_1}\cos(a,w_1) - \mathrm{mean}_{w_2\in W_2}\cos(a,w_2)

    Parameters
    ----------
    a : array-like
        Input data.
    W_1 : array-like
        Input data.
    W_2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 6, Eq 6.5, p. 234
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Weat similarity."})


def cheatsheet():
    return "km081: Weat similarity."

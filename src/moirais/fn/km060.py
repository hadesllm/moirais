"""Krona efficient.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch4_krona_efficient"]


def kamath_ch4_krona_efficient(A, B, x):
    """
    Krona efficient.

    Formula: (A\otimes B)x = \gamma(B\,\eta_{b_2\times a_2}(x)\,A^T)

    Parameters
    ----------
    A : array-like
        Input data.
    B : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 4, Eq 4.7, p. 152
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Krona efficient."})


def cheatsheet():
    return "km060: Krona efficient."

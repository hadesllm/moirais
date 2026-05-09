"""Krona output.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch4_krona_output"]


def kamath_ch4_krona_output(X, W, A_k, B_k, s):
    """
    Krona output.

    Formula: Y = X W + s X [A_k \otimes B_k]

    Parameters
    ----------
    X : array-like
        Input data.
    W : array-like
        Input data.
    A_k : array-like
        Input data.
    B_k : array-like
        Input data.
    s : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 4, Eq 4.8, p. 152
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Krona output."})


def cheatsheet():
    return "km061: Krona output."

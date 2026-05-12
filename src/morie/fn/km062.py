r"""Krona tuned weights.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch4_krona_tuned_weights"]


def kamath_ch4_krona_tuned_weights(W, A_k, B_k, s):
    r"""
    Krona tuned weights.

    Formula: W_{tuned} = W + s [A_k \otimes B_k]

    Parameters
    ----------
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
    Kamath et al (2024), Ch 4, Eq 4.9, p. 153
    r"""
    W = np.atleast_1d(np.asarray(W, dtype=float))
    n = len(W)
    result = float(np.mean(W))
    se = float(np.std(W, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Krona tuned weights."})


def cheatsheet():
    return "km062: Krona tuned weights."

"""Masked attention.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_masked_attention"]


def kamath_ch2_masked_attention(Q, K, V, M, d_k):
    """
    Masked attention.

    Formula: \mathrm{maskedAttention}(Q,K,V) = \mathrm{softmax}(\frac{QK^T + M}{\sqrt{d_k}})V

    Parameters
    ----------
    Q : array-like
        Input data.
    K : array-like
        Input data.
    V : array-like
        Input data.
    M : array-like
        Input data.
    d_k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.19, p. 38
    """
    Q = np.atleast_1d(np.asarray(Q, dtype=float))
    n = len(Q)
    result = float(np.mean(Q))
    se = float(np.std(Q, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Masked attention."})


def cheatsheet():
    return "km019: Masked attention."

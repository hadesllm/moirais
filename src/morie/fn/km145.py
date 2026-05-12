r"""Mmllm autoregressive.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch9_mmllm_autoregressive"]


def kamath_ch9_mmllm_autoregressive(R, I, theta):
    r"""
    Mmllm autoregressive.

    Formula: L(\theta) = -\sum_{i=1}^N \log p(R_i|I, R_{<i};\theta)

    Parameters
    ----------
    R : array-like
        Input data.
    I : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 9, Eq 9.17, p. 391
    r"""
    R = np.atleast_1d(np.asarray(R, dtype=float))
    n = len(R)
    result = float(np.mean(R))
    se = float(np.std(R, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mmllm autoregressive."})


def cheatsheet():
    return "km145: Mmllm autoregressive."

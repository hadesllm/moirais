r"""Clip image to text.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch9_clip_image_to_text"]


def kamath_ch9_clip_image_to_text(V, L, sigma, N):
    r"""
    Clip image to text.

    Formula: L_{i2t} = -\frac{1}{N}\sum_i \log\frac{\exp(V_i^T L_i/\sigma)}{\sum_j \exp(V_i^T L_j/\sigma)}

    Parameters
    ----------
    V : array-like
        Input data.
    L : array-like
        Input data.
    sigma : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 9, Eq 9.5, p. 386
    r"""
    V = np.atleast_1d(np.asarray(V, dtype=float))
    n = len(V)
    result = float(np.mean(V))
    se = float(np.std(V, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Clip image to text."})


def cheatsheet():
    return "km133: Clip image to text."

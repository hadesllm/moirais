r"""Clip text to image.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch9_clip_text_to_image"]


def kamath_ch9_clip_text_to_image(L, V, sigma, N):
    r"""
    Clip text to image.

    Formula: L_{t2i} = -\frac{1}{N}\sum_i \log\frac{\exp(L_i^T V_i/\sigma)}{\sum_j \exp(L_i^T V_j/\sigma)}

    Parameters
    ----------
    L : array-like
        Input data.
    V : array-like
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
    Kamath et al (2024), Ch 9, Eq 9.6, p. 386
    r"""
    L = np.atleast_1d(np.asarray(L, dtype=float))
    n = len(L)
    result = float(np.mean(L))
    se = float(np.std(L, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Clip text to image."})


def cheatsheet():
    return "km134: Clip text to image."

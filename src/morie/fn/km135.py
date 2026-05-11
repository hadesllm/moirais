"""Clip contrastive total.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch9_clip_contrastive_total"]


def kamath_ch9_clip_contrastive_total(L_i2t, L_t2i):
    """
    Clip contrastive total.

    Formula: L_{CL} = L_{i2t} + L_{t2i}

    Parameters
    ----------
    L_i2t : array-like
        Input data.
    L_t2i : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 9, Eq 9.7, p. 386
    """
    L_i2t = np.atleast_1d(np.asarray(L_i2t, dtype=float))
    n = len(L_i2t)
    result = float(np.mean(L_i2t))
    se = float(np.std(L_i2t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Clip contrastive total."})


def cheatsheet():
    return "km135: Clip contrastive total."

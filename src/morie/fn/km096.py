"""Gender projection reg.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch6_gender_projection_reg"]


def kamath_ch6_gender_projection_reg(W_stereo, g):
    """
    Gender projection reg.

    Formula: R = \sum_{w\in W_{stereo}} \frac{g}{\|g\|} w^T

    Parameters
    ----------
    W_stereo : array-like
        Input data.
    g : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 6, Eq 6.20, p. 243
    """
    W_stereo = np.atleast_1d(np.asarray(W_stereo, dtype=float))
    n = len(W_stereo)
    result = float(np.mean(W_stereo))
    se = float(np.std(W_stereo, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gender projection reg."})


def cheatsheet():
    return "km096: Gender projection reg."

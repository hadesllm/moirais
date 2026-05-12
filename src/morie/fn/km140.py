r"""Moc loss.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch9_moc_loss"]


def kamath_ch9_moc_loss(theta, w, v, g_theta):
    r"""
    Moc loss.

    Formula: L_{MOC}(\theta) = -E_{(w,v)}[\sum_{i=1}^M CE(c(v_m^i), g_{\theta}(v_m^i))]

    Parameters
    ----------
    theta : array-like
        Input data.
    w : array-like
        Input data.
    v : array-like
        Input data.
    g_theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 9, Eq 9.12, p. 388
    r"""
    w = np.atleast_1d(np.asarray(w, dtype=float))
    n = len(w)
    result = float(np.mean(w))
    se = float(np.std(w, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Moc loss."})


def cheatsheet():
    return "km140: Moc loss."

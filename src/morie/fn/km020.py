"""Ssl loss.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_ssl_loss"]


def kamath_ch2_ssl_loss(L_PTi, lambda_i):
    """
    Ssl loss.

    Formula: L_{SSL} = \lambda_1 L_{PT_1} + \lambda_2 L_{PT_2} + \dots + \lambda_m L_{PT_m}

    Parameters
    ----------
    L_PTi : array-like
        Input data.
    lambda_i : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.20, p. 50
    """
    L_PTi = np.atleast_1d(np.asarray(L_PTi, dtype=float))
    n = len(L_PTi)
    result = float(np.mean(L_PTi))
    se = float(np.std(L_PTi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ssl loss."})


def cheatsheet():
    return "km020: Ssl loss."

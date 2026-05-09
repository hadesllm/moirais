"""Gedi combined loss.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch6_gedi_combined_loss"]


def kamath_ch6_gedi_combined_loss(L_g, L_d, lam):
    """
    Gedi combined loss.

    Formula: L_{gd} = \lambda L_g + (1-\lambda) L_d

    Parameters
    ----------
    L_g : array-like
        Input data.
    L_d : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 6, Eq 6.29, p. 254
    """
    L_g = np.atleast_1d(np.asarray(L_g, dtype=float))
    n = len(L_g)
    result = float(np.mean(L_g))
    se = float(np.std(L_g, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gedi combined loss."})


def cheatsheet():
    return "km105: Gedi combined loss."

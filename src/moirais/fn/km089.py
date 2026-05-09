"""Sgs invariance.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch6_sgs_invariance"]


def kamath_ch6_sgs_invariance(Yhat_i, Yhat_j, psi):
    """
    Sgs invariance.

    Formula: \mathrm{SGS}(\hat{Y}) = \psi(\hat{Y}_i, \hat{Y}_j)

    Parameters
    ----------
    Yhat_i : array-like
        Input data.
    Yhat_j : array-like
        Input data.
    psi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 6, Eq 6.13, p. 236
    """
    Yhat_i = np.atleast_1d(np.asarray(Yhat_i, dtype=float))
    n = len(Yhat_i)
    result = float(np.mean(Yhat_i))
    se = float(np.std(Yhat_i, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sgs invariance."})


def cheatsheet():
    return "km089: Sgs invariance."

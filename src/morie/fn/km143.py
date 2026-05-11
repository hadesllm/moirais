"""Fom loss.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch9_fom_loss"]


def kamath_ch9_fom_loss(r_i, t_i, R):
    """
    Fom loss.

    Formula: L_{FOM} = -E[\sum_{i=1}^R \log P[r_i,t_i]]

    Parameters
    ----------
    r_i : array-like
        Input data.
    t_i : array-like
        Input data.
    R : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 9, Eq 9.15, p. 390
    """
    r_i = np.atleast_1d(np.asarray(r_i, dtype=float))
    n = len(r_i)
    result = float(np.mean(r_i))
    se = float(np.std(r_i, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fom loss."})


def cheatsheet():
    return "km143: Fom loss."

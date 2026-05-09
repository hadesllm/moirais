"""Modality encoder.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch9_modality_encoder"]


def kamath_ch9_modality_encoder(I_X, ME_X):
    """
    Modality encoder.

    Formula: F_X = \mathrm{ME}_X(I_X)

    Parameters
    ----------
    I_X : array-like
        Input data.
    ME_X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 9, Eq 9.1, p. 378
    """
    I_X = np.atleast_1d(np.asarray(I_X, dtype=float))
    n = len(I_X)
    result = float(np.mean(I_X))
    se = float(np.std(I_X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Modality encoder."})


def cheatsheet():
    return "km129: Modality encoder."

"""Gpt combined obj.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_gpt_combined_obj"]


def kamath_ch2_gpt_combined_obj(L_1, L_2, lam):
    """
    Gpt combined obj.

    Formula: L_3 = L_2(C) + \lambda \cdot L_1(U)

    Parameters
    ----------
    L_1 : array-like
        Input data.
    L_2 : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.37, p. 70
    """
    L_1 = np.atleast_1d(np.asarray(L_1, dtype=float))
    n = len(L_1)
    result = float(np.mean(L_1))
    se = float(np.std(L_1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gpt combined obj."})


def cheatsheet():
    return "km037: Gpt combined obj."

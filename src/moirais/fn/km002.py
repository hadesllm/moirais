"""Context vector.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_context_vector"]


def kamath_ch2_context_vector(h_1_h_T):
    """
    Context vector.

    Formula: c = m(h_1, \dots, h_T)

    Parameters
    ----------
    h_1_h_T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.2, p. 30
    """
    h_1_h_T = np.atleast_1d(np.asarray(h_1_h_T, dtype=float))
    n = len(h_1_h_T)
    result = float(np.mean(h_1_h_T))
    se = float(np.std(h_1_h_T, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Context vector."})


def cheatsheet():
    return "km002: Context vector."

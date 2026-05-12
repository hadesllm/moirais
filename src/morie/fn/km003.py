r"""Context simplest.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_context_simplest"]


def kamath_ch2_context_simplest(h_T):
    r"""
    Context simplest.

    Formula: c = m(h_1, \dots, h_T) = h_T

    Parameters
    ----------
    h_T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.3, p. 30
    r"""
    h_T = np.atleast_1d(np.asarray(h_T, dtype=float))
    n = len(h_T)
    result = float(np.mean(h_T))
    se = float(np.std(h_T, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Context simplest."})


def cheatsheet():
    return "km003: Context simplest."

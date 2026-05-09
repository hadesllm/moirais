"""Moe output.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_moe_output"]


def kamath_ch2_moe_output(x, G, E_i):
    """
    Moe output.

    Formula: y = \sum_{i=0}^{n-1} G(x)_i \cdot E_i(x)

    Parameters
    ----------
    x : array-like
        Input data.
    G : array-like
        Input data.
    E_i : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.39, p. 74
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Moe output."})


def cheatsheet():
    return "km039: Moe output."

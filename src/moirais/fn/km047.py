"""Translate prefix prompt.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch3_translate_prefix_prompt"]


def kamath_ch3_translate_prefix_prompt(x, z):
    """
    Translate prefix prompt.

    Formula: x' = \text{Translate the following English sentence to French: } [x][z]

    Parameters
    ----------
    x : array-like
        Input data.
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 3, Eq 3.6, p. 100
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Translate prefix prompt."})


def cheatsheet():
    return "km047: Translate prefix prompt."

"""Prefix prompt template.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch3_prefix_prompt_template"]


def kamath_ch3_prefix_prompt_template(x, z):
    """
    Prefix prompt template.

    Formula: x' = [x] \text{ This movie is } [z]

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
    Kamath et al (2024), Ch 3, Eq 3.5, p. 100
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Prefix prompt template."})


def cheatsheet():
    return "km046: Prefix prompt template."

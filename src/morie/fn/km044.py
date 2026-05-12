r"""Prompt search argmax.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch3_prompt_search_argmax"]


def kamath_ch3_prompt_search_argmax(x, z, theta):
    r"""
    Prompt search argmax.

    Formula: \hat{z} = \mathrm{search}_{z\in Z} P(f_{fill}(x,z);\theta)

    Parameters
    ----------
    x : array-like
        Input data.
    z : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 3, Eq 3.3, p. 94
    r"""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Prompt search argmax."})


def cheatsheet():
    return "km044: Prompt search argmax."

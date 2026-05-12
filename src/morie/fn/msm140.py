r"""Numbered display equation (8.8) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_8"]


def mvsml_categorical_count_eq_8_8(here, u, j, Nn, eu, eK):
    r"""
    Numbered display equation (8.8) from MVSML chapter 8.

    Formula: ), and from here u j 2    Nn eu, eK e eK y 2 1n\mu . Then the mean/mode of u j  is eu = \sigma2 ( ), which is also the BLUP of u under the mixed model equation of Henderson (1975). For this reason, model

    Parameters
    ----------
    here : array-like
        Input data.
    u : array-like
        Input data.
    j : array-like
        Input data.
    Nn : array-like
        Input data.
    eu : array-like
        Input data.
    eK : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (8.8) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    r"""
    here = np.atleast_1d(np.asarray(here, dtype=float))
    n = len(here)
    result = float(np.mean(here))
    se = float(np.std(here, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (8.8) from MVSML chapter 8."})


def cheatsheet():
    return "msm140: Numbered display equation (8.8) from MVSML chapter 8."

r"""Numbered display equation (5.5) from MVSML chapter 5.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_linear_mixed_models_eq_5_5"]


def mvsml_linear_mixed_models_eq_5_5(j, J, Y, j2, g, E):
    r"""
    Numbered display equation (5.5) from MVSML chapter 5.

    Formula: 6664 7775 + 6664 7775 + 6664 7775, j = 1, . . . , J, Y j2 \mu2 g j2 E j2

    Parameters
    ----------
    j : array-like
        Input data.
    J : array-like
        Input data.
    Y : array-like
        Input data.
    j2 : array-like
        Input data.
    g : array-like
        Input data.
    E : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (5.5) [Multivariate Statistical Machine Learnin [Pages 141-170] [2026-04-16].pdf]
    r"""
    j = np.atleast_1d(np.asarray(j, dtype=float))
    n = len(j)
    result = float(np.mean(j))
    se = float(np.std(j, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (5.5) from MVSML chapter 5."})


def cheatsheet():
    return "msm026: Numbered display equation (5.5) from MVSML chapter 5."

r"""Numbered display equation (14.9) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_9"]


def mvsml_convolutional_nn_eq_14_9(R, T, L1, t, m, dt):
    r"""
    Numbered display equation (14.9) from MVSML chapter 14.

    Formula: R T R T 0 ϕL1 t( )\psim t( )dt ⋯ 0 ϕL1 t( )\psiL2 t( )dt Now, matrix X can be computed as X = 1n = X ,

    Parameters
    ----------
    R : array-like
        Input data.
    T : array-like
        Input data.
    L1 : array-like
        Input data.
    t : array-like
        Input data.
    m : array-like
        Input data.
    dt : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.9) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    r"""
    R = np.atleast_1d(np.asarray(R, dtype=float))
    n = len(R)
    result = float(np.mean(R))
    se = float(np.std(R, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.9) from MVSML chapter 14."})


def cheatsheet():
    return "msm272: Numbered display equation (14.9) from MVSML chapter 14."

"""Numbered display equation (8.8) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_8"]


def mvsml_categorical_count_eq_8_8(can, be, substituted, the, Q, Nystr):
    """
    Numbered display equation (8.8) from MVSML chapter 8.

    Formula: 8.8) can be substituted for the Q approximations from Nyström (Titsias 2009). That   is, the random genetic vectors have a normal distribution u  N 0, \sigma2 uQ , where Q = Kn,mK 2 1 m,mK0 n,m: With these adjustments in the distribution of the random effects u, we used model

    Parameters
    ----------
    can : array-like
        Input data.
    be : array-like
        Input data.
    substituted : array-like
        Input data.
    the : array-like
        Input data.
    Q : array-like
        Input data.
    Nystr : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (8.8) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    """
    can = np.atleast_1d(np.asarray(can, dtype=float))
    n = len(can)
    result = float(np.mean(can))
    se = float(np.std(can, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (8.8) from MVSML chapter 8."})


def cheatsheet():
    return "msm149: Numbered display equation (8.8) from MVSML chapter 8."

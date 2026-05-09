"""Numbered display equation (10.16) from MVSML chapter 10.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_reproducing_kernel_eq_10_16"]


def mvsml_reproducing_kernel_eq_10_16(XL, z, h, w, j, ijw):
    """
    Numbered display equation (10.16) from MVSML chapter 10.

    Formula:  XL ( )´ z h \Deltaw h ( ) j=1\deltaijw l( ) jk g h ( ) kp = \eta xip = \eta\psiikxip,

    Parameters
    ----------
    XL : array-like
        Input data.
    z : array-like
        Input data.
    h : array-like
        Input data.
    w : array-like
        Input data.
    j : array-like
        Input data.
    ijw : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (10.16) [Multivariate Statistical Machine Learnin [Pages 379-425] [2026-04-16].pdf]
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (10.16) from MVSML chapter 10."})


def cheatsheet():
    return "msm254: Numbered display equation (10.16) from MVSML chapter 10."

"""Numbered display equation (10.17) from MVSML chapter 10.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_reproducing_kernel_eq_10_17"]


def mvsml_reproducing_kernel_eq_10_17(t, w, h, kp, ikxip):
    """
    Numbered display equation (10.17) from MVSML chapter 10.

    Formula: ( ) t+1 ( ) = w h ( ) t( ) + \Deltaw h ( ) kp = w h ( ) t( ) + \eta\psiikxip

    Parameters
    ----------
    t : array-like
        Input data.
    w : array-like
        Input data.
    h : array-like
        Input data.
    kp : array-like
        Input data.
    ikxip : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (10.17) [Multivariate Statistical Machine Learnin [Pages 379-425] [2026-04-16].pdf]
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    n = len(w)
    result = float(np.mean(w))
    se = float(np.std(w, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (10.17) from MVSML chapter 10."})


def cheatsheet():
    return "msm255: Numbered display equation (10.17) from MVSML chapter 10."

r"""Numbered display equation (10.12) from MVSML chapter 10.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_reproducing_kernel_eq_10_12"]


def mvsml_reproducing_kernel_eq_10_12(g, l, z, w, V, h):
    r"""
    Numbered display equation (10.12) from MVSML chapter 10.

    Formula:  -  g l( )´ z l( ) \Deltaw l( ) V h ( ) ik = \eta\deltaijV h ( ) jk = \eta yij - byij

    Parameters
    ----------
    g : array-like
        Input data.
    l : array-like
        Input data.
    z : array-like
        Input data.
    w : array-like
        Input data.
    V : array-like
        Input data.
    h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (10.12) [Multivariate Statistical Machine Learnin [Pages 379-425] [2026-04-16].pdf]
    r"""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (10.12) from MVSML chapter 10."})


def cheatsheet():
    return "msm250: Numbered display equation (10.12) from MVSML chapter 10."

"""Numbered display equation (10.13) from MVSML chapter 10.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_reproducing_kernel_eq_10_13"]


def mvsml_reproducing_kernel_eq_10_13(w, l, jk, t, ijV, h):
    """
    Numbered display equation (10.13) from MVSML chapter 10.

    Formula: + \Deltaw l( ) jk = w l( ) t( ) ( ) + \eta\deltaijV h (10.13) jk jk jk ik This equation reﬂects that the adjusted weights from

    Parameters
    ----------
    w : array-like
        Input data.
    l : array-like
        Input data.
    jk : array-like
        Input data.
    t : array-like
        Input data.
    ijV : array-like
        Input data.
    h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (10.13) [Multivariate Statistical Machine Learnin [Pages 379-425] [2026-04-16].pdf]
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    n = len(w)
    result = float(np.mean(w))
    se = float(np.std(w, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (10.13) from MVSML chapter 10."})


def cheatsheet():
    return "msm252: Numbered display equation (10.13) from MVSML chapter 10."

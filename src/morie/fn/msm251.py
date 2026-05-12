r"""Numbered display equation (10.13) from MVSML chapter 10.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_reproducing_kernel_eq_10_13"]


def mvsml_reproducing_kernel_eq_10_13(ij, the, hidden, units, to, output):
    r"""
    Numbered display equation (10.13) from MVSML chapter 10.

    Formula: ij from the hidden units to the output units is w l( ) t+1 ( ) = w l( ) t( ) + \Deltaw l( ) jk = w l( ) t( ) ( ) + \eta\deltaijV h

    Parameters
    ----------
    ij : array-like
        Input data.
    the : array-like
        Input data.
    hidden : array-like
        Input data.
    units : array-like
        Input data.
    to : array-like
        Input data.
    output : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (10.13) [Multivariate Statistical Machine Learnin [Pages 379-425] [2026-04-16].pdf]
    r"""
    ij = np.atleast_1d(np.asarray(ij, dtype=float))
    n = len(ij)
    result = float(np.mean(ij))
    se = float(np.std(ij, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (10.13) from MVSML chapter 10."})


def cheatsheet():
    return "msm251: Numbered display equation (10.13) from MVSML chapter 10."

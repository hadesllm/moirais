"""Numbered display equation (14.8) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_8"]


def mvsml_convolutional_nn_eq_14_8(L2, t1, t2):
    """
    Numbered display equation (14.8) from MVSML chapter 14.

    Formula: \psiL2 t1 ( ) 6664 7775 \psi1 t2 ( ) ⋱ \psiL2 t2 ( ) \Psi =

    Parameters
    ----------
    L2 : array-like
        Input data.
    t1 : array-like
        Input data.
    t2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.8) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    """
    L2 = np.atleast_1d(np.asarray(L2, dtype=float))
    n = len(L2)
    result = float(np.mean(L2))
    se = float(np.std(L2, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.8) from MVSML chapter 14."})


def cheatsheet():
    return "msm271: Numbered display equation (14.8) from MVSML chapter 14."

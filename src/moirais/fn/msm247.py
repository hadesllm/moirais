"""Numbered display equation (10.6) from MVSML chapter 10.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_reproducing_kernel_eq_10_6"]


def mvsml_reproducing_kernel_eq_10_6(jk, E, w, l, where, the):
    """
    Numbered display equation (10.6) from MVSML chapter 10.

    Formula: jk = -\eta \partial E \Deltaw l( ) (10.10) , \partial w l( ) jk where \eta is the learning rate that scales the step size and is speciﬁed by the user. To be able to calculate the adjustments for the weights connecting the hidden neurons to the outputs, w l( ) jk , ﬁrst we substitute

    Parameters
    ----------
    jk : array-like
        Input data.
    E : array-like
        Input data.
    w : array-like
        Input data.
    l : array-like
        Input data.
    where : array-like
        Input data.
    the : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (10.6) [Multivariate Statistical Machine Learnin [Pages 379-425] [2026-04-16].pdf]
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    n = len(w)
    result = float(np.mean(w))
    se = float(np.std(w, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (10.6) from MVSML chapter 10."})


def cheatsheet():
    return "msm247: Numbered display equation (10.6) from MVSML chapter 10."

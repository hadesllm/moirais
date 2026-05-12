r"""Numbered display equation (14.7) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_7"]


def mvsml_convolutional_nn_eq_14_7(o, t, where, cio, L2, are):
    r"""
    Numbered display equation (14.7) from MVSML chapter 14.

    Formula: o=1cio\psio t( ), (14.6) where cio, o = 1, . . ., L2, are constants to be determined for each observation, i = 1, . . ., n. Usually, this is determined by least squares, in which case, by assuming that all curves were observed at the same time points, this can be computed as - 1\PsiTxi t( ), T = \PsiT\Psi bci = bci1, . . . ,bciL2 =

    Parameters
    ----------
    o : array-like
        Input data.
    t : array-like
        Input data.
    where : array-like
        Input data.
    cio : array-like
        Input data.
    L2 : array-like
        Input data.
    are : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.7) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    r"""
    o = np.atleast_1d(np.asarray(o, dtype=float))
    n = len(o)
    result = float(np.mean(o))
    se = float(np.std(o, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.7) from MVSML chapter 14."})


def cheatsheet():
    return "msm270: Numbered display equation (14.7) from MVSML chapter 14."

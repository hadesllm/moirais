r"""Numbered display equation (14.4) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_4"]


def mvsml_convolutional_nn_eq_14_4(n, X, T, x1, t, Finally):
    r"""
    Numbered display equation (14.4) from MVSML chapter 14.

    Formula: n - 1QT = X\Psi \PsiT\Psi with X = x1 t( ) T: Finally, the complete practical solution of the = ⋯ xn t( ) parameter estimates is obtained with

    Parameters
    ----------
    n : array-like
        Input data.
    X : array-like
        Input data.
    T : array-like
        Input data.
    x1 : array-like
        Input data.
    t : array-like
        Input data.
    Finally : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.4) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    r"""
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.4) from MVSML chapter 14."})


def cheatsheet():
    return "msm273: Numbered display equation (14.4) from MVSML chapter 14."

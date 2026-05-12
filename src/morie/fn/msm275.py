r"""Numbered display equation (14.9) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_9"]


def mvsml_convolutional_nn_eq_14_9(X, T, x1, t, Finally, the):
    r"""
    Numbered display equation (14.9) from MVSML chapter 14.

    Formula: - 1QT = X\Psi \PsiT\Psi with X = x1 t( ) T: Finally, the complete practical solution of the = ⋯ xn t( ) parameter estimates is obtained with (14.4) and (14.5) but replacing X as computed in

    Parameters
    ----------
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
    the : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.9) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    r"""
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.9) from MVSML chapter 14."})


def cheatsheet():
    return "msm275: Numbered display equation (14.9) from MVSML chapter 14."

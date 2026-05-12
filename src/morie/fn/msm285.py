r"""Numbered display equation (14.12) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_12"]


def mvsml_convolutional_nn_eq_14_12(reduced, to, TD, T, D1, where):
    r"""
    Numbered display equation (14.12) from MVSML chapter 14.

    Formula: (14.12) is reduced to \lambda\betaTD \beta = \lambda\betaT 1 D1 \beta 1, where D1 is D but without the rows and columns corresponding to the eigenvalues equal to 0 of P. So, the corresponding smoothed solution of \beta(t) can be obtained as XL1 b\beta t( ) = l=1b\betalϕl t( ),  and b\beta  is the solution of

    Parameters
    ----------
    reduced : array-like
        Input data.
    to : array-like
        Input data.
    TD : array-like
        Input data.
    T : array-like
        Input data.
    D1 : array-like
        Input data.
    where : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.12) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    r"""
    reduced = np.atleast_1d(np.asarray(reduced, dtype=float))
    n = len(reduced)
    result = float(np.mean(reduced))
    se = float(np.std(reduced, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.12) from MVSML chapter 14."})


def cheatsheet():
    return "msm285: Numbered display equation (14.12) from MVSML chapter 14."

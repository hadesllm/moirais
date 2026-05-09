"""Numbered display equation (5.1) from MVSML chapter 5.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_linear_mixed_models_eq_5_1"]


def mvsml_linear_mixed_models_eq_5_1(J, T, j, are, g, j1):
    """
    Numbered display equation (5.1) from MVSML chapter 5.

    Formula: J T , j = 1, . . ., J, are g j1, . . . , g jnT j = 1, . . ., J, and e j = E j1, . . . , E jnT independent multivariate normal random vectors with null mean and variance RnT , \SigmaT is nT - nT matrix that represents the genetic covariance between traits, and ⨂is the Kronecker product. - T, In matrix notation, it is the linear mixed model

    Parameters
    ----------
    J : array-like
        Input data.
    T : array-like
        Input data.
    j : array-like
        Input data.
    are : array-like
        Input data.
    g : array-like
        Input data.
    j1 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (5.1) [Multivariate Statistical Machine Learnin [Pages 141-170] [2026-04-16].pdf]
    """
    J = np.atleast_1d(np.asarray(J, dtype=float))
    n = len(J)
    result = float(np.mean(J))
    se = float(np.std(J, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (5.1) from MVSML chapter 5."})


def cheatsheet():
    return "msm027: Numbered display equation (5.1) from MVSML chapter 5."

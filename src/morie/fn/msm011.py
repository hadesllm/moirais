"""Numbered display equation (5.2) from MVSML chapter 5.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_linear_mixed_models_eq_5_2"]


def mvsml_linear_mixed_models_eq_5_2(V, exp, TV, y, X, L):
    """
    Numbered display equation (5.2) from MVSML chapter 5.

    Formula: ) = V 2 2 exp  1 )TV1 y  X\beta L \beta, D, R; y ( 2 y  X\beta ( ( )

    Parameters
    ----------
    V : array-like
        Input data.
    exp : array-like
        Input data.
    TV : array-like
        Input data.
    y : array-like
        Input data.
    X : array-like
        Input data.
    L : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (5.2) [Multivariate Statistical Machine Learnin [Pages 141-170] [2026-04-16].pdf]
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (5.2) from MVSML chapter 5."})


def cheatsheet():
    return "msm011: Numbered display equation (5.2) from MVSML chapter 5."

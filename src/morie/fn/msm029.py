r"""Numbered display equation (5.5) from MVSML chapter 5.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_linear_mixed_models_eq_5_5"]


def mvsml_linear_mixed_models_eq_5_5(J, N, G, T, Similarly, the):
    r"""
    Numbered display equation (5.5) from MVSML chapter 5.

    Formula: J N 0, G⨂\SigmaT ( ): Similarly, the extended model that arises by adding more ﬁxed effects (X) can be speciﬁed by adding a term X\beta to the predictor: Y = 1IJ⨂InT ( )\mu + X\beta + Zb + e (5.5a) When \SigmaT and R are diagonal matrices, model

    Parameters
    ----------
    J : array-like
        Input data.
    N : array-like
        Input data.
    G : array-like
        Input data.
    T : array-like
        Input data.
    Similarly : array-like
        Input data.
    the : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (5.5) [Multivariate Statistical Machine Learnin [Pages 141-170] [2026-04-16].pdf]
    r"""
    J = np.atleast_1d(np.asarray(J, dtype=float))
    n = len(J)
    result = float(np.mean(J))
    se = float(np.std(J, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (5.5) from MVSML chapter 5."})


def cheatsheet():
    return "msm029: Numbered display equation (5.5) from MVSML chapter 5."

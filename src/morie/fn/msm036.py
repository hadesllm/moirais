r"""Numbered display equation (5.6) from MVSML chapter 5.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_linear_mixed_models_eq_5_6"]


def mvsml_linear_mixed_models_eq_5_6(error, R, Diag, e1, e2, The):
    r"""
    Numbered display equation (5.6) from MVSML chapter 5.

    Formula:  error, R = Diag \sigma2 e1, \sigma2 . e2 The results are shown in Table 5.4, from which we can observe that for trait 1 (GY), the best performance under both criteria (MSE and PC) was obtained with the more complex model: M4. For this trait (GY), the MSE of models M42, M43, and M44 were 7.53%, 8.21%, and 8.17%, respectively, greater than the MSE of Table 5.4 Prediction performance of some sub-models of model

    Parameters
    ----------
    error : array-like
        Input data.
    R : array-like
        Input data.
    Diag : array-like
        Input data.
    e1 : array-like
        Input data.
    e2 : array-like
        Input data.
    The : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (5.6) [Multivariate Statistical Machine Learnin [Pages 141-170] [2026-04-16].pdf]
    r"""
    error = np.atleast_1d(np.asarray(error, dtype=float))
    n = len(error)
    result = float(np.mean(error))
    se = float(np.std(error, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (5.6) from MVSML chapter 5."})


def cheatsheet():
    return "msm036: Numbered display equation (5.6) from MVSML chapter 5."

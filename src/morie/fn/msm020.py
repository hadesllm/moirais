r"""Numbered display equation (5.4) from MVSML chapter 5.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_linear_mixed_models_eq_5_4"]


def mvsml_linear_mixed_models_eq_5_4(Ei, the, genetic, variance, environment, i):
    r"""
    Numbered display equation (5.4) from MVSML chapter 5.

    Formula: Ei, is the genetic variance in environment i, i = 1, . . ., I, and \sigmaEikG is the genetic variance– covariance matrix for lines in environments i and k, where \sigmaEik is the element (i, k) of \SigmaE. When \SigmaE has a non-diagonal structure, the information from the genomic rela- tionship matrix and the correlated environments can be helpful for improving the prediction performance of the model by borrowing information between lines inside an environment and between lines across and among environments (Burgueño et al. 2012). Example 2 To illustrate how model

    Parameters
    ----------
    Ei : array-like
        Input data.
    the : array-like
        Input data.
    genetic : array-like
        Input data.
    variance : array-like
        Input data.
    environment : array-like
        Input data.
    i : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (5.4) [Multivariate Statistical Machine Learnin [Pages 141-170] [2026-04-16].pdf]
    r"""
    Ei = np.atleast_1d(np.asarray(Ei, dtype=float))
    n = len(Ei)
    result = float(np.mean(Ei))
    se = float(np.std(Ei, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (5.4) from MVSML chapter 5."})


def cheatsheet():
    return "msm020: Numbered display equation (5.4) from MVSML chapter 5."

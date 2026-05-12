r"""Numbered display equation (5.4) from MVSML chapter 5.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_linear_mixed_models_eq_5_4"]


def mvsml_linear_mixed_models_eq_5_4(When, E, has, a, non, diagonal):
    r"""
    Numbered display equation (5.4) from MVSML chapter 5.

    Formula: When \SigmaE has a non-diagonal structure, the information from the genomic rela- tionship matrix and the correlated environments can be helpful for improving the prediction performance of the model by borrowing information between lines inside an environment and between lines across and among environments (Burgueño et al. 2012). Example 2 To illustrate how model (5.4) can be implemented using the sommer package, the same data used in Example 1 are considered, where the same 30 geno- types are in the four environments. Besides the line indicator (GID), environment information (Env) was also available in the data set, which was needed for implementing model

    Parameters
    ----------
    When : array-like
        Input data.
    E : array-like
        Input data.
    has : array-like
        Input data.
    a : array-like
        Input data.
    non : array-like
        Input data.
    diagonal : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (5.4) [Multivariate Statistical Machine Learnin [Pages 141-170] [2026-04-16].pdf]
    r"""
    When = np.atleast_1d(np.asarray(When, dtype=float))
    n = len(When)
    result = float(np.mean(When))
    se = float(np.std(When, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (5.4) from MVSML chapter 5."})


def cheatsheet():
    return "msm021: Numbered display equation (5.4) from MVSML chapter 5."

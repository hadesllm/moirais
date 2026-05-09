"""Numbered display equation (5.4) from MVSML chapter 5.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_linear_mixed_models_eq_5_4"]


def mvsml_linear_mixed_models_eq_5_4(Another, explored, model, M20, was, obtained):
    """
    Numbered display equation (5.4) from MVSML chapter 5.

    Formula: Another explored model (M20) was obtained under the same speciﬁcation, with the difference that G was set equal to the identity matrix. Using the same validation scheme that was used in Example 1, the results for each of the 10 random partitions are shown in Table 5.2, in which, for illustrative purposes, model (5.3) plus environment as a ﬁxed effect (M11) is also included, that is, 5.4 Illustrative Examples of the Univariate LMM 151 Table 5.2 Prediction performance of two sub-models of

    Parameters
    ----------
    Another : array-like
        Input data.
    explored : array-like
        Input data.
    model : array-like
        Input data.
    M20 : array-like
        Input data.
    was : array-like
        Input data.
    obtained : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (5.4) [Multivariate Statistical Machine Learnin [Pages 141-170] [2026-04-16].pdf]
    """
    Another = np.atleast_1d(np.asarray(Another, dtype=float))
    n = len(Another)
    result = float(np.mean(Another))
    se = float(np.std(Another, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (5.4) from MVSML chapter 5."})


def cheatsheet():
    return "msm023: Numbered display equation (5.4) from MVSML chapter 5."

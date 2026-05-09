"""Numbered display equation (1.3) from MVSML chapter 1.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_general_eq_1_3"]


def mvsml_general_eq_1_3(environment, which, can, be, attributed, to):
    """
    Numbered display equation (1.3) from MVSML chapter 1.

    Formula: environment, which can be attributed to the fact that this model ignores the envi- ronmental effect and was only ﬁtted as a single-mean model, which implies that the environmental effects are included in the residuals. For this reason, we then incor- porated the environmental effect in the model as a separate effect. This ﬁxed-effects model is equal to a one-way classiﬁcation model as 16 1 General Elements of Genomic Selection and Statistical Learning GYij = \betai + eij, i = 1, ::, 5, j = 1, 2, 3,

    Parameters
    ----------
    environment : array-like
        Input data.
    which : array-like
        Input data.
    can : array-like
        Input data.
    be : array-like
        Input data.
    attributed : array-like
        Input data.
    to : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (1.3) [Multivariate Statistical Machine Learnin [Pages 1-34] [2026-04-16].pdf]
    """
    environment = np.atleast_1d(np.asarray(environment, dtype=float))
    n = len(environment)
    result = float(np.mean(environment))
    se = float(np.std(environment, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (1.3) from MVSML chapter 1."})


def cheatsheet():
    return "msm003: Numbered display equation (1.3) from MVSML chapter 1."

"""Numbered display equation (1.2) from MVSML chapter 1.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_general_eq_1_2"]


def mvsml_general_eq_1_2(Data, a, one, way, classi, cation):
    """
    Numbered display equation (1.2) from MVSML chapter 1.

    Formula: Data from a one-way classiﬁcation like the one given in Table 1.1 can be analyzed under both approaches with ﬁxed effects or random effects. The decision on which approach to use depends basically on the goal of the study, since if the goal is to make inferences about the population for which these environments (levels) were drawn, then the random effect approach is the best option, but if the goal is to make inferences about the particular environments (levels) selected in this experiment, then the ﬁxed-effects approach should be preferred. Assuming a simple model that ignores the environment GYij = \beta + eij, i = 1, ::, 5, j = 1, 2, 3,

    Parameters
    ----------
    Data : array-like
        Input data.
    a : array-like
        Input data.
    one : array-like
        Input data.
    way : array-like
        Input data.
    classi : array-like
        Input data.
    cation : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (1.2) [Multivariate Statistical Machine Learnin [Pages 1-34] [2026-04-16].pdf]
    """
    Data = np.atleast_1d(np.asarray(Data, dtype=float))
    n = len(Data)
    result = float(np.mean(Data))
    se = float(np.std(Data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (1.2) from MVSML chapter 1."})


def cheatsheet():
    return "msm002: Numbered display equation (1.2) from MVSML chapter 1."

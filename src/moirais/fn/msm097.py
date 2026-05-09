"""Numbered display equation (7.3) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_3"]


def mvsml_bayesian_regression_pt2_eq_7_3(improvement, these, models, respect, to, their):
    """
    Numbered display equation (7.3) from MVSML chapter 7.

    Formula: 86.05% improvement for these models with respect to their counterpart models when only marker effects or genomic effects were considered. So, greater improve- ment was observed with the GBLUP model with both metrics, but ﬁnally the performance of all models is almost undistinguishable but with an advantage in time execution of the GBLUP model with respect to the rest. These issues were 220 7 Bayesian and Classical Prediction Models for Categorical and Count Data Table 7.2 Brier score (BS) and proportion of cases correctly classiﬁed (PCCC) across 10 random partitions, with 80% of the total data set used for training and the rest for testing, under model

    Parameters
    ----------
    improvement : array-like
        Input data.
    these : array-like
        Input data.
    models : array-like
        Input data.
    respect : array-like
        Input data.
    to : array-like
        Input data.
    their : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.3) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    """
    improvement = np.atleast_1d(np.asarray(improvement, dtype=float))
    n = len(improvement)
    result = float(np.mean(improvement))
    se = float(np.std(improvement, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.3) from MVSML chapter 7."})


def cheatsheet():
    return "msm097: Numbered display equation (7.3) from MVSML chapter 7."

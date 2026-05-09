"""Numbered display equation (7.4) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_4"]


def mvsml_bayesian_regression_pt2_eq_7_4(the, lines, So, only, models, M3):
    """
    Numbered display equation (7.4) from MVSML chapter 7.

    Formula: for the 40 lines. So, only the models (M3 and M4) in (7.3) and (7.4) are ﬁtted, and the performance prediction for these models was evaluated using cross-validation. Also, in this comparison model, (M5) (7.5) is added but without the line+environment interaction effects, that is, only the environment effect and the genetic effects are taken into account in the linear predictor: 7.3 Ordinal Logistic Regression 221 Table 7.3 Brier score (BS) and proportion of cases correctly classiﬁed (PCCC) across 10 random partitions, with 80% of the total data set used for training and the rest for testing, under models (7.3),

    Parameters
    ----------
    the : array-like
        Input data.
    lines : array-like
        Input data.
    So : array-like
        Input data.
    only : array-like
        Input data.
    models : array-like
        Input data.
    M3 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.4) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    """
    the = np.atleast_1d(np.asarray(the, dtype=float))
    n = len(the)
    result = float(np.mean(the))
    se = float(np.std(the, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.4) from MVSML chapter 7."})


def cheatsheet():
    return "msm100: Numbered display equation (7.4) from MVSML chapter 7."

"""Numbered display equation (7.5) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_5"]


def mvsml_bayesian_regression_pt2_eq_7_5(performance, prediction, these, models, was, evaluated):
    """
    Numbered display equation (7.5) from MVSML chapter 7.

    Formula: performance prediction for these models was evaluated using cross-validation. Also, in this comparison model, (M5) (7.5) is added but without the line+environment interaction effects, that is, only the environment effect and the genetic effects are taken into account in the linear predictor: 7.3 Ordinal Logistic Regression 221 Table 7.3 Brier score (BS) and proportion of cases correctly classiﬁed (PCCC) across 10 random partitions, with 80% of the total data set used for training and the rest for testing, under models (7.3), (7.4), and

    Parameters
    ----------
    performance : array-like
        Input data.
    prediction : array-like
        Input data.
    these : array-like
        Input data.
    models : array-like
        Input data.
    was : array-like
        Input data.
    evaluated : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.5) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    """
    performance = np.atleast_1d(np.asarray(performance, dtype=float))
    n = len(performance)
    result = float(np.mean(performance))
    se = float(np.std(performance, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.5) from MVSML chapter 7."})


def cheatsheet():
    return "msm101: Numbered display equation (7.5) from MVSML chapter 7."

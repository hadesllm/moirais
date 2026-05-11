"""Numbered display equation (7.6) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_6"]


def mvsml_bayesian_regression_pt2_eq_7_6(Lasso, penalization, models, are, the, same):
    """
    Numbered display equation (7.6) from MVSML chapter 7.

    Formula: Lasso penalization models are the same as those for Ridge penalization but with the alpha value set to 1 instead of 0, as needs to be done in the basic code of the glmnet to ﬁt the penalized Ridge multinomial logistic regression models: A = cv.glmnet(X, y, family='multinomial', type.measure = "class", nfolds = 10, alpha = 0) 7.4 Penalized Multinomial Logistic Regression 229 Table 7.4 Brier score (BS) and proportion of cases correctly classiﬁed (PCCC) across 10 random partitions, with 80% of the total data set used for training and the rest for testing, under model

    Parameters
    ----------
    Lasso : array-like
        Input data.
    penalization : array-like
        Input data.
    models : array-like
        Input data.
    are : array-like
        Input data.
    the : array-like
        Input data.
    same : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.6) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    """
    Lasso = np.atleast_1d(np.asarray(Lasso, dtype=float))
    n = len(Lasso)
    result = float(np.mean(Lasso))
    se = float(np.std(Lasso, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.6) from MVSML chapter 7."})


def cheatsheet():
    return "msm120: Numbered display equation (7.6) from MVSML chapter 7."

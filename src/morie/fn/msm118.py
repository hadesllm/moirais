r"""Numbered display equation (7.7) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_7"]


def mvsml_bayesian_regression_pt2_eq_7_7(Like, the, penalized, logistic, regression, studied):
    r"""
    Numbered display equation (7.7) from MVSML chapter 7.

    Formula: . Like the penalized logistic regression studied in Chap. 3, the more j=1 \betacj common approach for choosing the “optimal” regularization parameter \lambda in the penalized multinomial regression model in (7.7) is by using a k-fold cross-validation strategy with misclassiﬁcation error as metrics. This will be used here. For more details, see Friedman et al. (2010). It is important to point out that the tuning parameter \lambda used in glmnet is equal to the one used in the penalized log-likelihood

    Parameters
    ----------
    Like : array-like
        Input data.
    the : array-like
        Input data.
    penalized : array-like
        Input data.
    logistic : array-like
        Input data.
    regression : array-like
        Input data.
    studied : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.7) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    r"""
    Like = np.atleast_1d(np.asarray(Like, dtype=float))
    n = len(Like)
    result = float(np.mean(Like))
    se = float(np.std(Like, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.7) from MVSML chapter 7."})


def cheatsheet():
    return "msm118: Numbered display equation (7.7) from MVSML chapter 7."

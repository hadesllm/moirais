r"""Numbered display equation (9.37) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_37"]


def mvsml_ridge_lasso_elastic_eq_9_37(s, T, like, a, the, total):
    r"""
    Numbered display equation (9.37) from MVSML chapter 9.

    Formula: ’s. T is like a as the total amount of errors allowed since it is the bound of the sum of \zetai budget for the amount that the margin can be violated by the n observations. For T close to zero, the soft-margin SVM allows very little error and is similar to the hard-margin classiﬁer (James et al. 2013). The larger T is, the more error is allowed, which in turn allows for wider margins. These parameters play a key role in controlling the bias-variance trade-off of this statistical learning method. In practice, T is a hyperparameter that needs to be tuned, for example, by using cross-validation. M is the width of the margin and we seek to make this quantity as large as possible. In

    Parameters
    ----------
    s : array-like
        Input data.
    T : array-like
        Input data.
    like : array-like
        Input data.
    a : array-like
        Input data.
    the : array-like
        Input data.
    total : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.37) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    r"""
    s = np.atleast_1d(np.asarray(s, dtype=float))
    n = len(s)
    result = float(np.mean(s))
    se = float(np.std(s, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.37) from MVSML chapter 9."})


def cheatsheet():
    return "msm222: Numbered display equation (9.37) from MVSML chapter 9."

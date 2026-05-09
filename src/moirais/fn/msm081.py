"""Numbered display equation (6.9) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_9"]


def mvsml_bayesian_regression_eq_6_9(internally, to, sample, vec, b2, Example):
    """
    Numbered display equation (6.9) from MVSML chapter 6.

    Formula: internally to sample from vec(b2). Example 4 To illustrate how to implement this model with the BMTME R package, we considered the data in Example 2, but now the explored model includes the trait– genotype–environment interaction. The average results of the prediction performance in terms of PC and MSE for implementing the same ﬁve-fold cross-validation used in Example 3 are shown in Table 6.5. These results show an improvement in terms of prediction performance with this model in all trait environments combinations and in both criteria (PC and MSE) to measure the prediction performance, except in trait MIXTIM and Env 2, where the MSE is slightly greater than the one obtained with model

    Parameters
    ----------
    internally : array-like
        Input data.
    to : array-like
        Input data.
    sample : array-like
        Input data.
    vec : array-like
        Input data.
    b2 : array-like
        Input data.
    Example : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.9) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    """
    sample = np.atleast_1d(np.asarray(sample, dtype=float))
    n = len(sample)
    result = float(np.mean(sample))
    se = float(np.std(sample, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.9) from MVSML chapter 6."})


def cheatsheet():
    return "msm081: Numbered display equation (6.9) from MVSML chapter 6."

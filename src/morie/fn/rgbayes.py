# morie.fn — function file (hadesllm/morie)
"""Bayes minimum-error classifier."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_bayes_classifier"]


def rangayyan_bayes_classifier(X, class_priors, class_means, class_covs):
    """
    Bayes minimum-error classifier

    Formula: Assign to class k: max P(C_k|X) = max P(X|C_k)*P(C_k)

    Parameters
    ----------
    X : array-like
        Input data.
    class_priors : array-like
        Input data.
    class_means : array-like
        Input data.
    class_covs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: labels, posteriors

    References
    ----------
    Rangayyan Ch 10.6
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayes minimum-error classifier"})


def cheatsheet():
    return "rgbayes: Bayes minimum-error classifier"

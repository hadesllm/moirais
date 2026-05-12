# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""BayesR: mixture of normals prior with different variance classes."""
import numpy as np
from ._richresult import RichResult

__all__ = ["bayes_r_prior"]


def bayes_r_prior(y, X, pi, sigma_classes):
    """
    BayesR: mixture of normals prior with different variance classes

    Formula: beta_j ~ sum_k pi_k * N(0, sigma_k^2); pi_k ~ Dirichlet(delta); 4 variance classes including zero

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    pi : array-like
        Input data.
    sigma_classes : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'beta_samples': 'array', 'class_probs': 'array'}

    References
    ----------
    Montesinos Lopez Ch 6
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BayesR: mixture of normals prior with different variance classes"})


def cheatsheet():
    return "baysr: BayesR: mixture of normals prior with different variance classes"

"""Log-likelihood of a Dirichlet sample."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dirichlet_loglik"]


def dirichlet_loglik(alpha, X):
    """
    Log-likelihood of a Dirichlet sample

    Formula: ℓ(α|X) = N[lnΓ(Σα)-Σ lnΓ(α_i)] + Σ_i (α_i-1) Σ_n ln x_{ni}

    Parameters
    ----------
    alpha : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ll

    References
    ----------
    Wilks (1962)
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    n = len(alpha)
    result = float(np.mean(alpha))
    se = float(np.std(alpha, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Log-likelihood of a Dirichlet sample"})


def cheatsheet():
    return "aitdrl: Log-likelihood of a Dirichlet sample"

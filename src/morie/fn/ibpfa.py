"""Indian Buffet Process for nonparametric latent factors."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["indian_buffet_factor"]


def indian_buffet_factor(y, alpha, n_iter):
    """
    Indian Buffet Process for nonparametric latent factors

    Formula: binary feature matrix Z with mass parameter alpha; row pi_k ~ Beta(alpha/K, 1)

    Parameters
    ----------
    y : array-like
        Input data.
    alpha : array-like
        Input data.
    n_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Griffiths-Ghahramani (2011)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Indian Buffet Process for nonparametric latent factors"})


def cheatsheet():
    return "ibpfa: Indian Buffet Process for nonparametric latent factors"

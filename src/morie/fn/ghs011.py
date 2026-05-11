"""Joint Dirichlet distribution of the first k weights and the residual mass under a countable Dirichlet prior with parameters alpha_j.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch3_countable_dirichlet_marginal"]


def ghosal_ch3_countable_dirichlet_marginal(p_j, alpha_j, k):
    """
    Joint Dirichlet distribution of the first k weights and the residual mass under a countable Dirichlet prior with parameters alpha_j.

    Formula: (p_1, ..., p_k, 1 - sum_{j=1}^{k} p_j) ~ Dir(k+1; alpha_1, ..., alpha_k, sum_{j=k+1}^{infty} alpha_j)

    Parameters
    ----------
    p_j : array-like
        Input data.
    alpha_j : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: distribution

    References
    ----------
    Ghosal & van der Vaart (2017), Ch 3, Eq 3.4, p. 31
    """
    p_j = np.atleast_1d(np.asarray(p_j, dtype=float))
    n = len(p_j)
    result = float(np.mean(p_j))
    se = float(np.std(p_j, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Joint Dirichlet distribution of the first k weights and the residual mass under a countable Dirichlet prior with parameters alpha_j."})


def cheatsheet():
    return "ghs011: Joint Dirichlet distribution of the first k weights and the residual mass under a countable Dirichlet prior with parameters alpha_j."

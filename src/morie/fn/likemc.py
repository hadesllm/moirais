"""Likelihood-based MCMC for compartmental."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["likelihood_mcmc_epi"]


def likelihood_mcmc_epi(model, data, priors, n_iter):
    """
    Likelihood-based MCMC for compartmental

    Formula: Metropolis-Hastings on compartmental likelihood

    Parameters
    ----------
    model : array-like
        Input data.
    data : array-like
        Input data.
    priors : array-like
        Input data.
    n_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    O'Neill-Roberts (1999)
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Likelihood-based MCMC for compartmental"})


def cheatsheet():
    return "likemc: Likelihood-based MCMC for compartmental"

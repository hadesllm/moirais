"""MCMC standard error of posterior mean."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mcmc_standard_error"]


def mcmc_standard_error(chains):
    """
    MCMC standard error of posterior mean

    Formula: MCSE = sd / sqrt(ESS)

    Parameters
    ----------
    chains : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Geyer (1992); Flegal et al. (2008)
    """
    chains = np.atleast_1d(np.asarray(chains, dtype=float))
    n = len(chains)
    result = float(np.mean(chains))
    se = float(np.std(chains, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MCMC standard error of posterior mean"})


def cheatsheet():
    return "mcsec: MCMC standard error of posterior mean"

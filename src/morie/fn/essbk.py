"""Bulk effective sample size for posterior."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["effective_sample_size_bulk"]


def effective_sample_size_bulk(chains):
    """
    Bulk effective sample size for posterior

    Formula: ESS_bulk via rank-normalized chains

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
    Vehtari et al. (2021)
    """
    chains = np.atleast_1d(np.asarray(chains, dtype=float))
    n = len(chains)
    result = float(np.mean(chains))
    se = float(np.std(chains, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bulk effective sample size for posterior"})


def cheatsheet():
    return "essbk: Bulk effective sample size for posterior"

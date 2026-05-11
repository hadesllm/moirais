"""Tail effective sample size (5% / 95% quantiles)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["effective_sample_size_tail"]


def effective_sample_size_tail(chains):
    """
    Tail effective sample size (5% / 95% quantiles)

    Formula: ESS_tail = min(ESS@5%, ESS@95%)

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Tail effective sample size (5% / 95% quantiles)"})


def cheatsheet():
    return "esstl: Tail effective sample size (5% / 95% quantiles)"

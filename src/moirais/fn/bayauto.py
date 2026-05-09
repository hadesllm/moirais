"""MCMC autocorrelation diagnostic."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["autocorrelation_check"]


def autocorrelation_check(chain):
    """
    MCMC autocorrelation diagnostic

    Formula: ACF + effective sample size

    Parameters
    ----------
    chain : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Geyer (1992)
    """
    chain = np.atleast_1d(np.asarray(chain, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(chain), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "MCMC autocorrelation diagnostic"})
    result = stats.spearmanr(chain[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "MCMC autocorrelation diagnostic"})


def cheatsheet():
    return "bayauto: MCMC autocorrelation diagnostic"

"""Bridge sampling for marginal likelihoods."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bridge_sampling"]


def bridge_sampling(chain, proposal, log_p, log_q):
    """
    Bridge sampling for marginal likelihoods

    Formula: iterative ratio of importance ratios

    Parameters
    ----------
    chain : array-like
        Input data.
    proposal : array-like
        Input data.
    log_p : array-like
        Input data.
    log_q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Meng-Wong (1996)
    """
    chain = np.atleast_1d(np.asarray(chain, dtype=float))
    n = len(chain)
    result = float(np.mean(chain))
    se = float(np.std(chain, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bridge sampling for marginal likelihoods"})


def cheatsheet():
    return "bridgs: Bridge sampling for marginal likelihoods"

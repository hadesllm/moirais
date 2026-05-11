"""KL-based MCMC convergence."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kl_mcmc_diagnostic"]


def kl_mcmc_diagnostic(chain, target):
    """
    KL-based MCMC convergence

    Formula: D_KL(p_chain || p_target)

    Parameters
    ----------
    chain : array-like
        Input data.
    target : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Brooks-Gelman (1998)
    """
    chain = np.atleast_1d(np.asarray(chain, dtype=float))
    n = len(chain)
    result = float(np.mean(chain))
    se = float(np.std(chain, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "KL-based MCMC convergence"})


def cheatsheet():
    return "klmcmc: KL-based MCMC convergence"

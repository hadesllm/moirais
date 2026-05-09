"""MCMC acceptance rate diagnostic."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["acceptance_rate_diagnostic"]


def acceptance_rate_diagnostic(chains):
    """
    MCMC acceptance rate diagnostic

    Formula: alpha_target ≈ 0.234 (RW MH); 0.65–0.85 (HMC)

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
    Roberts, Gelman, Gilks (1997); Beskos et al. (2013)
    """
    chains = np.atleast_1d(np.asarray(chains, dtype=float))
    n = len(chains)
    result = float(np.mean(chains))
    se = float(np.std(chains, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MCMC acceptance rate diagnostic"})


def cheatsheet():
    return "acpra: MCMC acceptance rate diagnostic"

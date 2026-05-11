"""E-BFMI (Bayesian fraction of missing information)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["energy_bayesian_fmi"]


def energy_bayesian_fmi(chains):
    """
    E-BFMI (Bayesian fraction of missing information)

    Formula: E-BFMI = Var(E_chain) / mean(Var_within E)

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
    Betancourt (2016) energy diagnostic
    """
    chains = np.atleast_1d(np.asarray(chains, dtype=float))
    n = len(chains)
    result = float(np.mean(chains))
    se = float(np.std(chains, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "E-BFMI (Bayesian fraction of missing information)"})


def cheatsheet():
    return "ebfmi: E-BFMI (Bayesian fraction of missing information)"

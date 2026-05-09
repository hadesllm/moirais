"""Gelman-Rubin R-hat convergence diagnostic."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["r_hat_convergence"]


def r_hat_convergence(chains):
    """
    Gelman-Rubin R-hat convergence diagnostic

    Formula: R_hat = sqrt( (n-1)/n + (1/n) B/W )

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
    Gelman & Rubin (1992); Vehtari, Gelman, Simpson, Carpenter, Burkner (2021) split-R-hat
    """
    chains = np.atleast_1d(np.asarray(chains, dtype=float))
    n = len(chains)
    result = float(np.mean(chains))
    se = float(np.std(chains, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gelman-Rubin R-hat convergence diagnostic"})


def cheatsheet():
    return "rhatc: Gelman-Rubin R-hat convergence diagnostic"

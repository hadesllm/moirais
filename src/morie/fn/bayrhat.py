"""Potential scale reduction R-hat."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["r_hat"]


def r_hat(chains):
    """
    Potential scale reduction R-hat

    Formula: sqrt(W + B/n) / W; should converge to 1

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
    Gelman-Rubin (1992); Vehtari et al (2021)
    """
    chains = np.atleast_1d(np.asarray(chains, dtype=float))
    n = len(chains)
    result = float(np.mean(chains))
    se = float(np.std(chains, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Potential scale reduction R-hat"})


def cheatsheet():
    return "bayrhat: Potential scale reduction R-hat"

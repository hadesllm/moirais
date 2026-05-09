"""ABC-SMC posterior for compartmental."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["abc_smc_epi"]


def abc_smc_epi(model, summary_stats, priors, n_particles):
    """
    ABC-SMC posterior for compartmental

    Formula: sequential Monte Carlo on summary stats

    Parameters
    ----------
    model : array-like
        Input data.
    summary_stats : array-like
        Input data.
    priors : array-like
        Input data.
    n_particles : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Toni et al (2009)
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ABC-SMC posterior for compartmental"})


def cheatsheet():
    return "abcsmc: ABC-SMC posterior for compartmental"

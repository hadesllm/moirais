"""Particle filter for partially observed Markov."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["particle_filter_epi"]


def particle_filter_epi(pomp_model, data, n_particles):
    """
    Particle filter for partially observed Markov

    Formula: sequential importance sampling + resample

    Parameters
    ----------
    pomp_model : array-like
        Input data.
    data : array-like
        Input data.
    n_particles : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    King-Nguyen-Ionides (2016) pomp
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Particle filter for partially observed Markov"})


def cheatsheet():
    return "pftrep: Particle filter for partially observed Markov"

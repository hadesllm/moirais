"""Deterministic seeded RNG factory for reproducible bootstrap."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_rng_seeded"]


def boot_rng_seeded(seed):
    """
    Deterministic seeded RNG factory for reproducible bootstrap

    Formula: rng = numpy.random.default_rng(seed)

    Parameters
    ----------
    seed : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rng

    References
    ----------
    PCG64 (O'Neill 2014)
    """
    seed = np.atleast_1d(np.asarray(seed, dtype=float))
    n = len(seed)
    result = float(np.mean(seed))
    se = float(np.std(seed, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Deterministic seeded RNG factory for reproducible bootstrap"})


def cheatsheet():
    return "btrnd: Deterministic seeded RNG factory for reproducible bootstrap"

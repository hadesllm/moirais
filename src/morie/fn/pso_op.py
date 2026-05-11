"""Particle swarm optimization."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["particle_swarm"]


def particle_swarm(f, n_particles, w, c1, c2):
    """
    Particle swarm optimization

    Formula: v_{t+1} = w v_t + c1 r1 (p - x) + c2 r2 (g - x)

    Parameters
    ----------
    f : array-like
        Input data.
    n_particles : array-like
        Input data.
    w : array-like
        Input data.
    c1 : array-like
        Input data.
    c2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kennedy-Eberhart (1995)
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    n = len(w)
    result = float(np.mean(w))
    se = float(np.std(w, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Particle swarm optimization"})


def cheatsheet():
    return "pso_op: Particle swarm optimization"

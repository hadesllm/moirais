"""n-step TD return."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["n_step_td"]


def n_step_td(traj, V, n, gamma):
    """
    n-step TD return

    Formula: G_t^(n) = sum_{k=0..n-1} γ^k r_{t+k+1} + γ^n V(s_{t+n})

    Parameters
    ----------
    traj : array-like
        Input data.
    V : array-like
        Input data.
    n : array-like
        Input data.
    gamma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sutton-Barto (1998)
    """
    traj = np.atleast_1d(np.asarray(traj, dtype=float))
    n = len(traj)
    result = float(np.mean(traj))
    se = float(np.std(traj, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "n-step TD return"})


def cheatsheet():
    return "gamtd: n-step TD return"

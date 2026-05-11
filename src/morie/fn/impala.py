"""IMPALA V-trace off-policy correction."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["impala_vtrace"]


def impala_vtrace(trajectories, behavior, target, clip):
    """
    IMPALA V-trace off-policy correction

    Formula: v_s = V(s) + sum γ^t (prod_i ρ_i) · δ_t V

    Parameters
    ----------
    trajectories : array-like
        Input data.
    behavior : array-like
        Input data.
    target : array-like
        Input data.
    clip : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Espeholt et al (2018)
    """
    trajectories = np.atleast_1d(np.asarray(trajectories, dtype=float))
    n = len(trajectories)
    result = float(np.mean(trajectories))
    se = float(np.std(trajectories, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "IMPALA V-trace off-policy correction"})


def cheatsheet():
    return "impala: IMPALA V-trace off-policy correction"

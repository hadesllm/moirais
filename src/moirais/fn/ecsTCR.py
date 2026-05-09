"""Equilibrium climate sensitivity / transient climate response."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ecs_tcr"]


def ecs_tcr(model_run, CO2_traj):
    """
    Equilibrium climate sensitivity / transient climate response

    Formula: ΔT_2x at equilibrium / at 2× CO2 ramp

    Parameters
    ----------
    model_run : array-like
        Input data.
    CO2_traj : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Charney (1979); IPCC AR6
    """
    model_run = np.atleast_1d(np.asarray(model_run, dtype=float))
    n = len(model_run)
    result = float(np.mean(model_run))
    se = float(np.std(model_run, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Equilibrium climate sensitivity / transient climate response"})


def cheatsheet():
    return "ecsTCR: Equilibrium climate sensitivity / transient climate response"

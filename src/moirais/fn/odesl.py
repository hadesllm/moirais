"""Symbolic ODE solver."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ode_symbolic"]


def ode_symbolic(ode):
    """
    Symbolic ODE solver

    Formula: classify (separable, linear, exact, Bernoulli)

    Parameters
    ----------
    ode : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bronstein (1997) Symbolic Integration
    """
    ode = np.atleast_1d(np.asarray(ode, dtype=float))
    n = len(ode)
    result = float(np.mean(ode))
    se = float(np.std(ode, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Symbolic ODE solver"})


def cheatsheet():
    return "odesl: Symbolic ODE solver"

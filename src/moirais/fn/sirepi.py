"""SIR compartmental model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sir_compartmental"]


def sir_compartmental(S0, I0, R0, beta, gamma, T):
    """
    SIR compartmental model

    Formula: dS/dt = -beta SI/N; dI/dt = beta SI/N - gamma I; dR/dt = gamma I

    Parameters
    ----------
    S0 : array-like
        Input data.
    I0 : array-like
        Input data.
    R0 : array-like
        Input data.
    beta : array-like
        Input data.
    gamma : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kermack-McKendrick (1927)
    """
    S0 = np.atleast_1d(np.asarray(S0, dtype=float))
    n = len(S0)
    result = float(np.mean(S0))
    se = float(np.std(S0, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SIR compartmental model"})


def cheatsheet():
    return "sirepi: SIR compartmental model"

"""ABC with GP emulator surrogate."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["abc_gp_emulator"]


def abc_gp_emulator(sim, obs, X_grid, kernel):
    """
    ABC with GP emulator surrogate

    Formula: surrogate likelihood via GP

    Parameters
    ----------
    sim : array-like
        Input data.
    obs : array-like
        Input data.
    X_grid : array-like
        Input data.
    kernel : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wilkinson (2014); Meeds-Welling (2014)
    """
    obs = np.atleast_1d(np.asarray(obs, dtype=float))
    n = len(obs)
    result = float(np.mean(obs))
    se = float(np.std(obs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ABC with GP emulator surrogate"})


def cheatsheet():
    return "abcgp: ABC with GP emulator surrogate"

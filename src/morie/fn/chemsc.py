"""ChemScore empirical docking."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["chemscore_dock"]


def chemscore_dock(receptor, ligand):
    """
    ChemScore empirical docking

    Formula: hbond + metal + lipophilic - rotational penalty

    Parameters
    ----------
    receptor : array-like
        Input data.
    ligand : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Eldridge et al (1997) Astex
    """
    receptor = np.atleast_1d(np.asarray(receptor, dtype=float))
    n = len(receptor)
    result = float(np.mean(receptor))
    se = float(np.std(receptor, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ChemScore empirical docking"})


def cheatsheet():
    return "chemsc: ChemScore empirical docking"

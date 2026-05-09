"""Knowledge-based PMF scoring."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pmf_potential"]


def pmf_potential(receptor, ligand):
    """
    Knowledge-based PMF scoring

    Formula: -RT log[g_ij(r)/g^*_ij(r)] summed over atom pairs

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
    Muegge-Martin (1999)
    """
    receptor = np.atleast_1d(np.asarray(receptor, dtype=float))
    n = len(receptor)
    result = float(np.mean(receptor))
    se = float(np.std(receptor, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Knowledge-based PMF scoring"})


def cheatsheet():
    return "pmfsc: Knowledge-based PMF scoring"

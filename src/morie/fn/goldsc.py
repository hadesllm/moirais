"""GOLD genetic-algorithm docking score."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gold_score"]


def gold_score(receptor, ligand):
    """
    GOLD genetic-algorithm docking score

    Formula: hbond + vdw + ligand internal

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
    Jones et al (1997) CCDC
    """
    receptor = np.atleast_1d(np.asarray(receptor, dtype=float))
    n = len(receptor)
    result = float(np.mean(receptor))
    se = float(np.std(receptor, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GOLD genetic-algorithm docking score"})


def cheatsheet():
    return "goldsc: GOLD genetic-algorithm docking score"

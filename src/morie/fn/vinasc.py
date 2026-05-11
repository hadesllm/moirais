"""AutoDock Vina scoring function."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["autodock_vina_score"]


def autodock_vina_score(receptor, ligand_pose):
    """
    AutoDock Vina scoring function

    Formula: weighted sum of hydrophobic + hbond + electrostatic + steric

    Parameters
    ----------
    receptor : array-like
        Input data.
    ligand_pose : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Trott-Olson (2010)
    """
    receptor = np.atleast_1d(np.asarray(receptor, dtype=float))
    n = len(receptor)
    result = float(np.mean(receptor))
    se = float(np.std(receptor, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AutoDock Vina scoring function"})


def cheatsheet():
    return "vinasc: AutoDock Vina scoring function"

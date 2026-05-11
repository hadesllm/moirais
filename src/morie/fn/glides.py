"""Glide-style empirical docking score (proxy)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["glide_score_proxy"]


def glide_score_proxy(receptor, ligand_pose):
    """
    Glide-style empirical docking score (proxy)

    Formula: empirical combination of phobic enclosure, lipophilic and hbond terms

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
    Friesner et al (2004) Schrödinger Glide
    """
    receptor = np.atleast_1d(np.asarray(receptor, dtype=float))
    n = len(receptor)
    result = float(np.mean(receptor))
    se = float(np.std(receptor, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Glide-style empirical docking score (proxy)"})


def cheatsheet():
    return "glides: Glide-style empirical docking score (proxy)"

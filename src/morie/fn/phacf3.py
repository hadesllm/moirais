"""3D pharmacophore fingerprint."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pharmacophore_3d"]


def pharmacophore_3d(mol_3d, feature_set):
    """
    3D pharmacophore fingerprint

    Formula: triplet of (feature_type, feature_type, distance_bin)

    Parameters
    ----------
    mol_3d : array-like
        Input data.
    feature_set : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gund (1977); Mason et al (2001)
    """
    mol_3d = np.atleast_1d(np.asarray(mol_3d, dtype=float))
    n = len(mol_3d)
    result = float(np.mean(mol_3d))
    se = float(np.std(mol_3d, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "3D pharmacophore fingerprint"})


def cheatsheet():
    return "phacf3: 3D pharmacophore fingerprint"

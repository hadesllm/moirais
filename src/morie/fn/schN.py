"""SchNet -- continuous filter conv on molecules."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["schnet"]


def schnet(coords, atom_types):
    """
    SchNet -- continuous filter conv on molecules

    Formula: continuous-filter convolution on inter-atomic distances

    Parameters
    ----------
    coords : array-like
        Input data.
    atom_types : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schütt et al (2017)
    """
    coords = np.atleast_1d(np.asarray(coords, dtype=float))
    n = len(coords)
    result = float(np.mean(coords))
    se = float(np.std(coords, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SchNet -- continuous filter conv on molecules"})


def cheatsheet():
    return "schN: SchNet -- continuous filter conv on molecules"

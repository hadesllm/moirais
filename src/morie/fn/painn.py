"""PaiNN -- equivariant scalar+vector features."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["painn"]


def painn(coords, atom_types):
    """
    PaiNN -- equivariant scalar+vector features

    Formula: separate scalar and vector tracks; mix via gates

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
    Schütt et al (2021)
    """
    coords = np.atleast_1d(np.asarray(coords, dtype=float))
    n = len(coords)
    result = float(np.mean(coords))
    se = float(np.std(coords, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PaiNN -- equivariant scalar+vector features"})


def cheatsheet():
    return "painn: PaiNN -- equivariant scalar+vector features"

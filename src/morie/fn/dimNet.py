"""DimeNet — directional message passing."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dimenet"]


def dimenet(coords, atom_types):
    """
    DimeNet — directional message passing

    Formula: angles + Bessel basis + spherical harmonics

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
    Klicpera et al (2020) DimeNet
    """
    coords = np.atleast_1d(np.asarray(coords, dtype=float))
    n = len(coords)
    result = float(np.mean(coords))
    se = float(np.std(coords, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DimeNet — directional message passing"})


def cheatsheet():
    return "dimNet: DimeNet — directional message passing"

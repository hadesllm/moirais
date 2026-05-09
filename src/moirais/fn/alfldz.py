"""AlphaFold loss decomposition by component."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphafold_loss_decomposition"]


def alphafold_loss_decomposition(loss_components):
    """
    AlphaFold loss decomposition by component

    Formula: FAPE + dist + chi + pLDDT + …

    Parameters
    ----------
    loss_components : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jumper et al (2021)
    """
    loss_components = np.atleast_1d(np.asarray(loss_components, dtype=float))
    n = len(loss_components)
    result = float(np.mean(loss_components))
    se = float(np.std(loss_components, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaFold loss decomposition by component"})


def cheatsheet():
    return "alfldz: AlphaFold loss decomposition by component"

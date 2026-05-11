"""Population attributable fraction (PAF)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["population_attributable"]


def population_attributable(pe, RR):
    """
    Population attributable fraction (PAF)

    Formula: PAF = pe(RR-1)/(pe(RR-1)+1)

    Parameters
    ----------
    pe : array-like
        Input data.
    RR : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Levin (1953)
    """
    pe = np.atleast_1d(np.asarray(pe, dtype=float))
    n = len(pe)
    result = float(np.mean(pe))
    se = float(np.std(pe, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Population attributable fraction (PAF)"})


def cheatsheet():
    return "pareff: Population attributable fraction (PAF)"

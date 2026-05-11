"""Emissions inventory (sector × fuel × EF)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["emissions_inventory"]


def emissions_inventory(activity, ef):
    """
    Emissions inventory (sector × fuel × EF)

    Formula: E = sum activity · emission_factor

    Parameters
    ----------
    activity : array-like
        Input data.
    ef : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    IPCC GHG Inventory Guidelines
    """
    activity = np.atleast_1d(np.asarray(activity, dtype=float))
    n = len(activity)
    result = float(np.mean(activity))
    se = float(np.std(activity, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Emissions inventory (sector × fuel × EF)"})


def cheatsheet():
    return "airbed: Emissions inventory (sector × fuel × EF)"

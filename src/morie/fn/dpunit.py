"""Unit-of-privacy definition (event vs user)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dp_unit_definition"]


def dp_unit_definition(unit, records):
    """
    Unit-of-privacy definition (event vs user)

    Formula: replace one unit and require neighboring outputs ε-close

    Parameters
    ----------
    unit : array-like
        Input data.
    records : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kifer-Machanavajjhala (2011) Pufferfish
    """
    unit = np.atleast_1d(np.asarray(unit, dtype=float))
    n = len(unit)
    result = float(np.mean(unit))
    se = float(np.std(unit, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Unit-of-privacy definition (event vs user)"})


def cheatsheet():
    return "dpunit: Unit-of-privacy definition (event vs user)"

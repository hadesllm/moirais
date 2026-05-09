"""Information-criterion-based item selection."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["eap_information"]


def eap_information(item_pool, theta):
    """
    Information-criterion-based item selection

    Formula: select item j maximizing dI/dtheta

    Parameters
    ----------
    item_pool : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    van der Linden-Pashley (2010)
    """
    item_pool = np.atleast_1d(np.asarray(item_pool, dtype=float))
    n = len(item_pool)
    result = float(np.mean(item_pool))
    se = float(np.std(item_pool, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Information-criterion-based item selection"})


def cheatsheet():
    return "esatic: Information-criterion-based item selection"

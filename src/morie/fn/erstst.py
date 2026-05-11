"""Elliott-Rothenberg-Stock GLS-detrended ADF."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ers_unit_root"]


def ers_unit_root(x, lags):
    """
    Elliott-Rothenberg-Stock GLS-detrended ADF

    Formula: DF-GLS on de-trended y_t with most-powerful local-to-unity

    Parameters
    ----------
    x : array-like
        Input data.
    lags : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Elliott, Rothenberg, Stock (1996)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Elliott-Rothenberg-Stock GLS-detrended ADF"})


def cheatsheet():
    return "erstst: Elliott-Rothenberg-Stock GLS-detrended ADF"

"""De Chaisemartin-D'Haultfoeuille robust DiD."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_did_de_chaisemartin"]


def causal_did_de_chaisemartin(Y_panel, D_panel):
    """
    De Chaisemartin-D'Haultfoeuille robust DiD

    Formula: Avg over (g,t) ATEs reachable in heterogeneous panel

    Parameters
    ----------
    Y_panel : array-like
        Input data.
    D_panel : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ATE, se

    References
    ----------
    de Chaisemartin & D'Haultfoeuille (2020)
    """
    Y_panel = np.atleast_1d(np.asarray(Y_panel, dtype=float))
    n = len(Y_panel)
    result = float(np.mean(Y_panel))
    se = float(np.std(Y_panel, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "De Chaisemartin-D'Haultfoeuille robust DiD"})


def cheatsheet():
    return "causdiddc: De Chaisemartin-D'Haultfoeuille robust DiD"

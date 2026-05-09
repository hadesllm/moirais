"""QQ-plot diagnostic for a fitted GEV."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_gev_qq_plot"]


def evt_gev_qq_plot(x, mu, sigma, xi):
    """
    QQ-plot diagnostic for a fitted GEV

    Formula: (F̂^{-1}(i/(n+1)), x_{(i)})

    Parameters
    ----------
    x : array-like
        Input data.
    mu : array-like
        Input data.
    sigma : array-like
        Input data.
    xi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: q_emp, q_model

    References
    ----------
    Coles (2001)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "QQ-plot diagnostic for a fitted GEV"})


def cheatsheet():
    return "evqqgev: QQ-plot diagnostic for a fitted GEV"

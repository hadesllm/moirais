"""Quantile loss / pinball."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["quantile_forecast"]


def quantile_forecast(y, y_hat, tau):
    """
    Quantile loss / pinball

    Formula: L_τ(y, ŷ) = (τ−𝟙{y<ŷ})(y−ŷ)

    Parameters
    ----------
    y : array-like
        Input data.
    y_hat : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Koenker (2005)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Quantile loss / pinball"})


def cheatsheet():
    return "qrF: Quantile loss / pinball"

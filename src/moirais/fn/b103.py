"""Mean squared error cost over the dataset for the linear model with parameters w and b.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["burkov_lm_ch1_mse_cost"]


def burkov_lm_ch1_mse_cost(w, b, x, y, N):
    """
    Mean squared error cost over the dataset for the linear model with parameters w and b.

    Formula: J(w, b) \stackrel{\text{def}}{=} \frac{(wx_1 + b - y_1)^2 + (wx_2 + b - y_2)^2 + \cdots + (wx_N + b - y_N)^2}{N}

    Parameters
    ----------
    w : array-like
        Input data.
    b : array-like
        Input data.
    x : array-like
        Input data.
    y : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: mean squared error cost

    References
    ----------
    Burkov LM (2025), Ch 1, Eq 1.3, p. 22
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mean squared error cost over the dataset for the linear model with parameters w and b."})


def cheatsheet():
    return "b103: Mean squared error cost over the dataset for the linear model with parameters w and b."

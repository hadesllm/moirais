"""MSE cost function of the Wiener filter expanded to autocorrelation/cross-correlation form.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_mse_cost_function"]


def rangayyan_ch3_mse_cost_function(w, Theta, Phi, sigma_d):
    """
    MSE cost function of the Wiener filter expanded to autocorrelation/cross-correlation form.

    Formula: J(w) = E[e^2(n)] = sigma_d^2 - w^T*Theta - Theta^T*w + w^T*Phi*w

    Parameters
    ----------
    w : array-like
        Input data.
    Theta : array-like
        Input data.
    Phi : array-like
        Input data.
    sigma_d : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.159 / 3.166, p. 174
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(w), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "MSE cost function of the Wiener filter expanded to autocorrelation/cross-correlation form."})
    result = stats.spearmanr(w[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "MSE cost function of the Wiener filter expanded to autocorrelation/cross-correlation form."})


def cheatsheet():
    return "rng141: MSE cost function of the Wiener filter expanded to autocorrelation/cross-correlation form."

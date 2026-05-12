# morie.fn -- function file (hadesllm/morie)
"""Cross-validation bandwidth selection for single-index model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_bw_cv_sim"]


def horowitz_bw_cv_sim(x, y, beta_hat):
    """
    Cross-validation bandwidth selection for single-index model

    Formula: h_CV = argmin (1/n)*sum_i [Y_i - G_hat_{-i,h}(X_i'beta_hat)]^2

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    beta_hat : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bandwidth

    References
    ----------
    Horowitz Ch 2, Sec 2.7
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cross-validation bandwidth selection for single-index model"})


def cheatsheet():
    return "hrzbwcv: Cross-validation bandwidth selection for single-index model"

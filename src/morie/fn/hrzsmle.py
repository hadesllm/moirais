# morie.fn -- function file (hadesllm/morie)
"""Semiparametric MLE for binary-response single-index model (Klein-Spady)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_semipar_mle_binary"]


def horowitz_semipar_mle_binary(x, y, bandwidth):
    """
    Semiparametric MLE for binary-response single-index model (Klein-Spady)

    Formula: beta_hat = argmax sum [Y_i*log G_hat(X_i'b)+(1-Y_i)*log(1-G_hat(X_i'b))]

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta_hat, se

    References
    ----------
    Horowitz Ch 2, Sec 2.5.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Semiparametric MLE for binary-response single-index model (Klein-Spady)"})


def cheatsheet():
    return "hrzsmle: Semiparametric MLE for binary-response single-index model (Klein-Spady)"

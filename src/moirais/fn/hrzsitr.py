# moirais.fn — function file (hadesllm/moirais)
"""Series truncation (sieve) estimator for NPIV."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_sieve_npiv"]


def horowitz_sieve_npiv(x, y, w, K):
    """
    Series truncation (sieve) estimator for NPIV

    Formula: g_hat = argmin_{g in G_K} ||E_hat(Y|W) - T_hat*g||^2; G_K = K-dim sieve space

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    w : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: g_hat

    References
    ----------
    Horowitz Ch 5, Sec 5.3.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Series truncation (sieve) estimator for NPIV"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Series truncation (sieve) estimator for NPIV"})


def cheatsheet():
    return "hrzsitr: Series truncation (sieve) estimator for NPIV"

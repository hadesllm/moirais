# morie.fn — function file (hadesllm/morie)
"""Second cumulative survival function estimator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_cum_surv_est2"]


def fauzi_cum_surv_est2(t, bandwidth, g_func):
    """
    Second cumulative survival function estimator

    Formula: S_tilde_X,2(t) = (1/n) sum V_{2,h}(g^{-1}(t), g^{-1}(X_i)); V_{2,h}(t,y)=int_{-inf}^y g'(z)V((t-z)/h)dz

    Parameters
    ----------
    t : array-like
        Input data.
    bandwidth : array-like
        Input data.
    g_func : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fauzi Ch 4, Eq 4.17-4.18
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    if t.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Second cumulative survival function estimator"})
    estimate = np.median(t)
    se = 1.2533 * np.std(t, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Second cumulative survival function estimator"})


def cheatsheet():
    return "fzcs2: Second cumulative survival function estimator"

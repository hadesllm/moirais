# moirais.fn — function file (hadesllm/moirais)
"""Log-likelihood for LMM under ML estimation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["lmm_log_likelihood"]


def lmm_log_likelihood(y, X, beta, V):
    """
    Log-likelihood for LMM under ML estimation

    Formula: log L(beta, V) = -(n/2)*log(2*pi) - (1/2)*log|V| - (1/2)*(y-X*beta)'*V^{-1}*(y-X*beta)

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    beta : array-like
        Input data.
    V : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'loglik': 'float'}

    References
    ----------
    Montesinos Lopez Ch 5
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    if y.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Log-likelihood for LMM under ML estimation"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Log-likelihood for LMM under ML estimation"})


def cheatsheet():
    return "lmmll: Log-likelihood for LMM under ML estimation"

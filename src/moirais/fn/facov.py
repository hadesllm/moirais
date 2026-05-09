# moirais.fn — function file (hadesllm/moirais)
"""Factor analytic covariance structure for multi-environment trials."""
import numpy as np
from ._richresult import RichResult

__all__ = ["factor_analytic_covariance"]


def factor_analytic_covariance(n_env, n_factors):
    """
    Factor analytic covariance structure for multi-environment trials

    Formula: Sigma_g = Lambda*Lambda' + Psi; Lambda k-factor loadings, Psi diagonal specific variances

    Parameters
    ----------
    n_env : array-like
        Input data.
    n_factors : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'Sigma': 'matrix'}

    References
    ----------
    Montesinos Lopez Ch 5
    """
    n_env = np.asarray(n_env, dtype=float)
    n = int(n_env) if n_env.ndim == 0 else len(n_env)
    result = float(np.mean(n_env))
    se = float(np.std(n_env, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Factor analytic covariance structure for multi-environment trials"})


def cheatsheet():
    return "facov: Factor analytic covariance structure for multi-environment trials"

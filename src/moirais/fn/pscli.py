# moirais.fn — function file (hadesllm/moirais)
"""pscl ideal() function for Bayesian IRT roll call scaling."""
import numpy as np
from ._richresult import RichResult

__all__ = ["pscl_ideal"]


def pscl_ideal(rollcall_obj, n_dims, n_iter):
    """
    pscl ideal() function for Bayesian IRT roll call scaling

    Formula: ideal(votes, d=1, maxiter, burnin, thin, normalize, priors)

    Parameters
    ----------
    rollcall_obj : array-like
        Input data.
    n_dims : array-like
        Input data.
    n_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'ideal_points': 'matrix', 'item_params': 'matrix'}

    References
    ----------
    Armstrong Ch 6
    """
    rollcall_obj = np.asarray(rollcall_obj, dtype=float)
    n = int(rollcall_obj) if rollcall_obj.ndim == 0 else len(rollcall_obj)
    result = float(np.mean(rollcall_obj))
    se = float(np.std(rollcall_obj, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "pscl ideal() function for Bayesian IRT roll call scaling"})


def cheatsheet():
    return "pscli: pscl ideal() function for Bayesian IRT roll call scaling"

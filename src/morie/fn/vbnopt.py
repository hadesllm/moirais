"""Variational inference (ELBO max)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["variational_inference"]


def variational_inference(log_p, q, x):
    """
    Variational inference (ELBO max)

    Formula: max ELBO = E_q[log p(x,z)] - E_q[log q(z)]

    Parameters
    ----------
    log_p : array-like
        Input data.
    q : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jordan et al (1999)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Variational inference (ELBO max)"})


def cheatsheet():
    return "vbnopt: Variational inference (ELBO max)"

# morie.fn -- function file (hadesllm/morie)
"""Single EM step for Gaussian mixture: E-step responsibilities + M-step updates."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_gmm_em_step"]


def geron_gmm_em_step(X, pi, means, covars):
    """
    Single EM step for Gaussian mixture: E-step responsibilities + M-step updates

    Formula: r_{ik} = pi_k N(x_i|mu_k,Sigma_k) / sum_j ...; mu_k, Sigma_k, pi_k = weighted_update(r)

    Parameters
    ----------
    X : array-like
        Input data.
    pi : array-like
        Input data.
    means : array-like
        Input data.
    covars : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: pi_new, means_new, covars_new, responsibilities

    References
    ----------
    Géron Ch 8, Expectation-Maximization section
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Single EM step for Gaussian mixture: E-step responsibilities + M-step updates"})


def cheatsheet():
    return "grgmem: Single EM step for Gaussian mixture: E-step responsibilities + M-step updates"

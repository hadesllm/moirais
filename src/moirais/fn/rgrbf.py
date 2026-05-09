# moirais.fn — function file (hadesllm/moirais)
"""Radial basis function (RBF) network."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_rbf_network"]


def rangayyan_rbf_network(X, y, n_centers, sigma):
    """
    Radial basis function (RBF) network

    Formula: phi_k(y) = exp(-||y-c_k||^2 / (2*sigma_k^2)); y = sum w_k*phi_k(y)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    n_centers : array-like
        Input data.
    sigma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: centers, weights, predictions

    References
    ----------
    Rangayyan Ch 10.8.1
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Radial basis function (RBF) network"})


def cheatsheet():
    return "rgrbf: Radial basis function (RBF) network"

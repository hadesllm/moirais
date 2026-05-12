# morie.fn — function file (hadesllm/morie)
"""Christoffel symbols from metric tensor."""

__all__ = ["chrsf"]

import numpy as np


def chrsf(
    metric: np.ndarray,
    coords: np.ndarray = None,
    metric_derivs: np.ndarray = None,
    h: float = 1e-6,
    metric_func=None,
) -> dict:
    r"""
    Compute Christoffel symbols of the second kind.

    .. math::

        \\Gamma^\\lambda_{\\mu\\nu} = \\frac{1}{2} g^{\\lambda\\sigma}
        \\left( \\partial_\\mu g_{\\nu\\sigma}
              + \\partial_\\nu g_{\\mu\\sigma}
              - \\partial_\\sigma g_{\\mu\\nu} \\right)

    Parameters
    ----------
    metric : np.ndarray
        (n, n) metric tensor g_{mu nu}.
    coords : np.ndarray, optional
        Coordinate values (used with metric_func for numerical derivatives).
    metric_derivs : np.ndarray, optional
        (n, n, n) array where metric_derivs[sigma, mu, nu] = d_sigma g_{mu nu}.
        If None and metric_func is provided, computed numerically.
    h : float
        Step size for numerical differentiation.
    metric_func : callable, optional
        Function(coords) -> (n,n) metric. Used for numerical derivatives.

    Returns
    -------
    dict
        Keys: christoffel (n,n,n) array Gamma^lam_{mu,nu}, metric_inverse.
    """
    n = metric.shape[0]
    if metric.shape != (n, n):
        raise ValueError("metric must be square.")

    ginv = np.linalg.inv(metric)

    if metric_derivs is None:
        if metric_func is not None and coords is not None:
            coords = np.asarray(coords, dtype=float)
            dg = np.zeros((n, n, n))
            for sig in range(n):
                dx = np.zeros(n)
                dx[sig] = h
                gp = metric_func(coords + dx)
                gm = metric_func(coords - dx)
                dg[sig] = (gp - gm) / (2.0 * h)
        else:
            dg = np.zeros((n, n, n))
    else:
        dg = np.asarray(metric_derivs, dtype=float)

    Gamma = np.zeros((n, n, n))
    for lam in range(n):
        for mu in range(n):
            for nu in range(n):
                s = 0.0
                for sig in range(n):
                    s += 0.5 * ginv[lam, sig] * (
                        dg[mu, nu, sig] + dg[nu, mu, sig] - dg[sig, mu, nu]
                    )
                Gamma[lam, mu, nu] = s

    return {
        "christoffel": Gamma,
        "metric_inverse": ginv,
    }

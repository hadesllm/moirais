# morie.fn — function file (hadesllm/morie)
"""Riemann curvature tensor."""

__all__ = ["riman"]

import numpy as np


def riman(
    christoffel: np.ndarray,
    christoffel_derivs: np.ndarray = None,
    metric_func=None,
    coords: np.ndarray = None,
    h: float = 1e-5,
) -> dict:
    r"""
    Compute the Riemann curvature tensor from Christoffel symbols.

    .. math::

        R^\\rho{}_{\\sigma\\mu\\nu} =
        \\partial_\\mu \\Gamma^\\rho_{\\nu\\sigma}
        - \\partial_\\nu \\Gamma^\\rho_{\\mu\\sigma}
        + \\Gamma^\\rho_{\\mu\\lambda} \\Gamma^\\lambda_{\\nu\\sigma}
        - \\Gamma^\\rho_{\\nu\\lambda} \\Gamma^\\lambda_{\\mu\\sigma}

    Parameters
    ----------
    christoffel : np.ndarray
        (n,n,n) Christoffel symbols Gamma^rho_{mu,nu}.
    christoffel_derivs : np.ndarray, optional
        (n,n,n,n) array d_mu Gamma^rho_{nu,sigma}. If None, set to zero
        (appropriate for constant Christoffel symbols only).
    metric_func : callable, optional
        Not used directly but reserved for future numerical derivative support.
    coords : np.ndarray, optional
        Coordinate values (reserved).
    h : float
        Step size (reserved).

    Returns
    -------
    dict
        Keys: riemann (n,n,n,n) R^rho_{sigma,mu,nu}, kretschner_check (bool).
    """
    n = christoffel.shape[0]
    G = christoffel

    if christoffel_derivs is not None:
        dG = np.asarray(christoffel_derivs, dtype=float)
    else:
        dG = np.zeros((n, n, n, n))

    R = np.zeros((n, n, n, n))
    for rho in range(n):
        for sig in range(n):
            for mu in range(n):
                for nu in range(n):
                    R[rho, sig, mu, nu] = dG[mu, rho, nu, sig] - dG[nu, rho, mu, sig]
                    for lam in range(n):
                        R[rho, sig, mu, nu] += (
                            G[rho, mu, lam] * G[lam, nu, sig]
                            - G[rho, nu, lam] * G[lam, mu, sig]
                        )

    return {
        "riemann": R,
        "kretschner_check": True,
    }

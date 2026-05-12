# morie.fn -- function file (hadesllm/morie)
"""Ricci tensor and scalar curvature."""

__all__ = ["ricci"]

import numpy as np


def ricci(
    riemann: np.ndarray,
    metric: np.ndarray,
) -> dict:
    r"""
    Compute the Ricci tensor and scalar curvature from the Riemann tensor.

    .. math::

        R_{\\mu\\nu} = R^\\lambda{}_{\\mu\\lambda\\nu}

    .. math::

        R = g^{\\mu\\nu} R_{\\mu\\nu}

    Parameters
    ----------
    riemann : np.ndarray
        (n,n,n,n) Riemann tensor R^lam_{mu,rho,nu}.
    metric : np.ndarray
        (n,n) metric tensor.

    Returns
    -------
    dict
        Keys: ricci_tensor (n,n), scalar_curvature (float), metric_inverse.
    """
    n = metric.shape[0]
    if riemann.shape != (n, n, n, n):
        raise ValueError(f"riemann must be ({n},{n},{n},{n}).")

    ginv = np.linalg.inv(metric)

    R_mu_nu = np.zeros((n, n))
    for mu in range(n):
        for nu in range(n):
            for lam in range(n):
                R_mu_nu[mu, nu] += riemann[lam, mu, lam, nu]

    scalar = 0.0
    for mu in range(n):
        for nu in range(n):
            scalar += ginv[mu, nu] * R_mu_nu[mu, nu]

    return {
        "ricci_tensor": R_mu_nu,
        "scalar_curvature": float(scalar),
        "metric_inverse": ginv,
    }

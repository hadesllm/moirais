# morie.fn -- function file (hadesllm/morie)
"""Lorentz transformation (special relativity)."""

__all__ = ["lortz"]

import numpy as np


def lortz(
    event: np.ndarray,
    v: float,
    c: float = 299792458.0,
) -> dict:
    r"""
    Apply a Lorentz boost along the x-axis.

    Transforms a 4-vector :math:`(ct, x, y, z)` from frame S to S'
    moving at velocity *v* along x:

    .. math::

        \\begin{pmatrix} ct' \\\\ x' \\end{pmatrix}
        = \\begin{pmatrix} \\gamma & -\\beta\\gamma \\\\
           -\\beta\\gamma & \\gamma \\end{pmatrix}
        \\begin{pmatrix} ct \\\\ x \\end{pmatrix}

    where :math:`\\beta = v/c` and :math:`\\gamma = 1/\\sqrt{1-\\beta^2}`.

    Parameters
    ----------
    event : np.ndarray
        4-vector ``[ct, x, y, z]`` in the original frame.
    v : float
        Relative velocity of boosted frame (m/s).
    c : float
        Speed of light (default SI, m/s).

    Returns
    -------
    dict
        Keys: ``event_prime`` (transformed 4-vector), ``gamma``,
        ``beta``, ``boost_matrix`` (4x4 ndarray).
    """
    event = np.asarray(event, dtype=float)
    if event.shape != (4,):
        raise ValueError("event must be a length-4 array [ct, x, y, z].")
    if abs(v) >= c:
        raise ValueError("|v| must be < c.")

    beta = v / c
    gamma = 1.0 / np.sqrt(1.0 - beta ** 2)

    L = np.eye(4)
    L[0, 0] = gamma
    L[0, 1] = -beta * gamma
    L[1, 0] = -beta * gamma
    L[1, 1] = gamma

    event_prime = L @ event
    return {
        "event_prime": event_prime,
        "gamma": gamma,
        "beta": beta,
        "boost_matrix": L,
    }

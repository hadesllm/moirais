# morie.fn — function file (hadesllm/morie)
"""Basic reproduction number R0 from next-generation matrix."""

from __future__ import annotations

import numpy as np


def r_naught_ngm(
    F: np.ndarray,
    V: np.ndarray,
) -> dict:
    """Compute R0 as the spectral radius of the next-generation matrix FV^{-1}.

    .. math::

        R_0 = \\rho(F V^{-1})

    where F is the transmission matrix and V is the transition matrix,
    both evaluated at the disease-free equilibrium.

    Parameters
    ----------
    F : np.ndarray
        Transmission matrix (n x n), non-negative entries.
    V : np.ndarray
        Transition matrix (n x n), must be invertible.

    Returns
    -------
    dict
        Keys: 'R0', 'ngm' (next-generation matrix), 'eigenvalues'.

    Raises
    ------
    ValueError
        If matrices are not square or not the same size.
    np.linalg.LinAlgError
        If V is singular.

    References
    ----------
    Diekmann, O., Heesterbeek, J. A. P., & Metz, J. A. J. (1990). On the
    definition and the computation of the basic reproduction ratio R0.
    Journal of Mathematical Biology, 28(4), 365-382.

    van den Driessche, P. & Watmough, J. (2002). Reproduction numbers and
    sub-threshold endemic equilibria for compartmental models of disease
    transmission. Mathematical Biosciences, 180(1-2), 29-48.
    """
    F = np.asarray(F, dtype=float)
    V = np.asarray(V, dtype=float)

    if F.ndim != 2 or F.shape[0] != F.shape[1]:
        raise ValueError("F must be a square matrix.")
    if V.ndim != 2 or V.shape[0] != V.shape[1]:
        raise ValueError("V must be a square matrix.")
    if F.shape != V.shape:
        raise ValueError("F and V must have the same dimensions.")

    V_inv = np.linalg.inv(V)
    ngm = F @ V_inv
    eigenvalues = np.linalg.eigvals(ngm)
    r0 = float(np.max(np.abs(eigenvalues)))

    return {
        "R0": r0,
        "ngm": ngm,
        "eigenvalues": eigenvalues,
    }


rnaht = r_naught_ngm


def cheatsheet() -> str:
    return "r_naught_ngm({}) -> R0 from next-generation matrix (spectral radius)."

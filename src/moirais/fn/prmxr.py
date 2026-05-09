# moirais.fn — function file (hadesllm/moirais)
"""Promax rotation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult
from .vrimx import varimax


def promax(
    loadings: np.ndarray,
    power: int = 4,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> DescriptiveResult:
    """Promax (oblique) rotation: varimax followed by target oblique rotation.

    Parameters
    ----------
    loadings : ndarray (p, k)
        Unrotated loading matrix.
    power : int
        Promax power parameter (typically 2-4).
    max_iter : int
        Max iterations for varimax step.
    tol : float
        Convergence tolerance for varimax step.

    Returns
    -------
    DescriptiveResult
        ``value`` is the rotated loadings.
        ``extra`` has ``phi`` (factor correlations), ``rotation_matrix``.
    """
    A = np.asarray(loadings, dtype=np.float64)

    vr = varimax(A, max_iter=max_iter, tol=tol)
    V = vr.value

    target = np.sign(V) * np.abs(V) ** power
    T = np.linalg.lstsq(V, target, rcond=None)[0]

    col_norms = np.linalg.norm(T, axis=0)
    col_norms[col_norms == 0] = 1.0
    T /= col_norms

    rotated = V @ T
    T_inv = np.linalg.inv(T)
    phi = T_inv @ T_inv.T

    D = np.diag(1.0 / np.sqrt(np.diag(phi)))
    phi_corr = D @ phi @ D

    return DescriptiveResult(
        name="Promax",
        value=rotated,
        extra={
            "rotation_matrix": T,
            "phi": phi_corr,
            "varimax_loadings": V,
        },
    )


prmxr = promax


def cheatsheet() -> str:
    return "promax({}) -> Promax oblique rotation."

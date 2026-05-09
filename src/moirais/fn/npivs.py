# moirais.fn — function file (hadesllm/moirais)
"""Nonparametric IV via series (sieve) estimation.

Approximates the structural function :math:`h(x)` using a linear
combination of basis functions (Hermite polynomials or B-splines),
then estimates coefficients from the moment condition

.. math::

    E[Y - h(X) \\mid Z] = 0

via two-stage least squares with basis expansions.

References
----------
Newey, W. K. & Powell, J. L. (2003). Instrumental variable estimation
    of nonparametric models. *Econometrica*, 71(5), 1565--1578.
Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Chapter 7.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def npivs(
    Y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    *,
    n_basis_x: int = 6,
    n_basis_z: int = 8,
    basis: str = "polynomial",
) -> dict[str, Any]:
    r"""Nonparametric IV via sieve/series estimation.

    Parameters
    ----------
    Y : np.ndarray
        Outcome vector, shape ``(n,)``.
    X : np.ndarray
        Endogenous regressor, shape ``(n,)``.
    Z : np.ndarray
        Instrument, shape ``(n,)``.
    n_basis_x : int
        Number of basis functions for :math:`X`.
    n_basis_z : int
        Number of basis functions for :math:`Z`.
    basis : str
        ``"polynomial"`` or ``"cosine"``.

    Returns
    -------
    dict[str, Any]
        ``coefficients``, ``x_grid``, ``h_hat``, ``n``,
        ``n_basis_x``, ``n_basis_z``, ``method``.
    """
    Y = np.asarray(Y, dtype=float).ravel()
    X = np.asarray(X, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float).ravel()
    n = len(Y)
    if len(X) != n or len(Z) != n:
        raise ValueError("Y, X, Z must have the same length.")
    if n_basis_z < n_basis_x:
        raise ValueError("n_basis_z must be >= n_basis_x for identification.")

    def _build_basis(v: np.ndarray, k: int) -> np.ndarray:
        v_std = (v - v.mean()) / (v.std() + 1e-12)
        if basis == "cosine":
            return np.column_stack(
                [np.cos(np.pi * j * (v_std + 1) / 2) for j in range(k)]
            )
        return np.column_stack([v_std ** j for j in range(k)])

    Phi_x = _build_basis(X, n_basis_x)
    Phi_z = _build_basis(Z, n_basis_z)

    P_z = Phi_z @ np.linalg.lstsq(Phi_z, Phi_x, rcond=None)[0]

    coef = np.linalg.lstsq(P_z, Y, rcond=None)[0]

    x_grid = np.linspace(X.min(), X.max(), 100)
    Phi_grid = _build_basis(x_grid, n_basis_x)
    h_hat = Phi_grid @ coef

    return {
        "coefficients": coef,
        "x_grid": x_grid,
        "h_hat": h_hat,
        "n": n,
        "n_basis_x": n_basis_x,
        "n_basis_z": n_basis_z,
        "method": "NPIV_Sieve",
    }


npivs_fn = npivs


def cheatsheet() -> str:
    return "npivs(Y, X, Z) -> Nonparametric IV via sieve estimation (Newey & Powell 2003)."

"""Lagrange kriging system solver."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def lagrange_kriging_system(
    gamma_matrix: np.ndarray,
    gamma_0: np.ndarray,
) -> SpatialResult:
    r"""Solve the ordinary kriging system via Lagrange multipliers.

    .. math::

        \begin{bmatrix} \Gamma & \mathbf{1} \\
        \mathbf{1}^T & 0 \end{bmatrix}
        \begin{bmatrix} \boldsymbol{\lambda} \\ \mu \end{bmatrix}
        = \begin{bmatrix} \boldsymbol{\gamma}_0 \\ 1 \end{bmatrix}

    Parameters
    ----------
    gamma_matrix : np.ndarray
        Variogram matrix between observations, shape ``(n, n)``.
    gamma_0 : np.ndarray
        Variogram vector between obs and target, shape ``(n,)``.

    Returns
    -------
    SpatialResult
        ``statistic`` is the Lagrange multiplier.
        ``extra`` has ``weights``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 5.

    .. epigraph::

        "Keelah se'lai." -- Tali, Mass Effect
    """
    G = np.asarray(gamma_matrix, dtype=np.float64)
    g0 = np.asarray(gamma_0, dtype=np.float64).ravel()
    n = len(g0)

    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = G
    A[:n, n] = 1.0
    A[n, :n] = 1.0
    b = np.zeros(n + 1)
    b[:n] = g0
    b[n] = 1.0

    lam = np.linalg.solve(A, b)

    return SpatialResult(
        name="lagrange_kriging_system",
        statistic=float(lam[n]),
        p_value=None,
        extra={"weights": lam[:n], "full_solution": lam},
    )


sglag = lagrange_kriging_system


def cheatsheet() -> str:
    return "lagrange_kriging_system({}) -> Lagrange kriging system solver."

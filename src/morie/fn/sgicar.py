"""Intrinsic conditional autoregressive (ICAR) model."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def intrinsic_car_model(
    Z: np.ndarray,
    W: np.ndarray,
) -> SpatialResult:
    r"""Fit an intrinsic CAR (ICAR) model (:math:`\rho = 1`).

    The ICAR is improper: the precision matrix is
    :math:`\mathbf{Q} = \mathbf{D} - \mathbf{W}` (the graph Laplacian).

    Parameters
    ----------
    Z : np.ndarray
        Response, shape ``(n,)``.
    W : np.ndarray
        Adjacency matrix, shape ``(n, n)``.

    Returns
    -------
    SpatialResult
        ``statistic`` is the spatial smoothness measure
        :math:`Z^T Q Z / n`.
        ``extra`` has ``Q``, ``conditional_means``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 6.

    .. epigraph::

    """
    Z = np.asarray(Z, dtype=np.float64).ravel()
    W = np.asarray(W, dtype=np.float64)
    n = len(Z)

    D = np.diag(W.sum(axis=1))
    Q = D - W
    smoothness = float(Z @ Q @ Z / n)

    d = np.diag(D).copy()
    d[d == 0] = 1.0
    cond_means = W @ Z / d

    return SpatialResult(
        name="intrinsic_car_model",
        statistic=smoothness,
        p_value=None,
        extra={"Q": Q, "conditional_means": cond_means},
    )


sgicar = intrinsic_car_model


def cheatsheet() -> str:
    return "intrinsic_car_model({}) -> Intrinsic conditional autoregressive (ICAR) model."

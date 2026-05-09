"""Cholesky-based Gaussian random field simulation."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def cholesky_grf_sim(
    coords: np.ndarray,
    cov_model: str = "exponential",
    cov_params: dict | None = None,
    n_sims: int = 1,
    seed: int = 42,
) -> SpatialResult:
    r"""Simulate a Gaussian random field via Cholesky decomposition.

    .. math::

        \mathbf{Z} = \mathbf{L}\,\mathbf{w}, \quad
        \mathbf{w} \sim N(0, I)

    where :math:`\mathbf{C} = \mathbf{L}\mathbf{L}^T`.

    Parameters
    ----------
    coords : np.ndarray
        Simulation coordinates, shape ``(n, 2)``.
    cov_model : str
        ``"exponential"`` or ``"gaussian"``.
    cov_params : dict, optional
        ``{"sill", "range", "nugget"}``.
    n_sims : int
        Number of independent realizations.
    seed : int
        RNG seed.

    Returns
    -------
    SpatialResult
        ``statistic`` is mean of first realization.
        ``extra`` has ``simulations`` shape ``(n_sims, n)``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 7.

    .. epigraph::

        "To see a World in a Grain of Sand." -- Blake / Final Fantasy
    """
    rng = np.random.default_rng(seed)
    coords = np.asarray(coords, dtype=np.float64)
    n = len(coords)
    params = cov_params or {"sill": 1.0, "range": 1.0, "nugget": 0.0}
    sill = params.get("sill", 1.0)
    r = params.get("range", 1.0)
    nug = params.get("nugget", 0.0)

    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    if cov_model == "gaussian":
        C = sill * np.exp(-((dist / r) ** 2)) + nug * np.eye(n)
    else:
        C = sill * np.exp(-dist / r) + nug * np.eye(n)

    L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
    sims = np.empty((n_sims, n))
    for k in range(n_sims):
        sims[k] = L @ rng.standard_normal(n)

    return SpatialResult(
        name="cholesky_grf_sim",
        statistic=float(np.mean(sims[0])),
        p_value=None,
        extra={"simulations": sims},
    )


sgchol = cholesky_grf_sim


def cheatsheet() -> str:
    return "cholesky_grf_sim({}) -> Cholesky-based Gaussian random field simulation."

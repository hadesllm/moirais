"""Sequential Gaussian simulation."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def sequential_gaussian_sim(
    Z: np.ndarray,
    coords: np.ndarray,
    grid: np.ndarray,
    cov_model: str = "exponential",
    cov_params: dict | None = None,
    seed: int = 42,
    max_neighbours: int = 12,
) -> SpatialResult:
    r"""Sequential Gaussian simulation on a grid.

    Visits grid nodes in random order, kriging from conditioning
    data + previously simulated nodes, then draws from the
    kriging distribution.

    Parameters
    ----------
    Z : np.ndarray
        Conditioning data, shape ``(n,)``.
    coords : np.ndarray
        Conditioning coordinates, shape ``(n, 2)``.
    grid : np.ndarray
        Simulation grid, shape ``(m, 2)``.
    cov_model : str
        Covariance model.
    cov_params : dict, optional
        ``{"sill", "range", "nugget"}``.
    seed : int
        RNG seed.
    max_neighbours : int
        Max conditioning points per node.

    Returns
    -------
    SpatialResult
        ``statistic`` is mean of simulated field.
        ``extra`` has ``simulated_values``, ``grid``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 7.

    .. epigraph::

        "Eyes on me." -- FF8 OST
    """
    rng = np.random.default_rng(seed)
    Z = np.asarray(Z, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    n = len(Z)
    m = len(grid)
    params = cov_params or {"sill": 1.0, "range": 1.0, "nugget": 0.0}
    sill = params.get("sill", 1.0)
    r = params.get("range", 1.0)
    nug = params.get("nugget", 0.0)

    def _gamma(h):
        return nug + sill * (1.0 - np.exp(-h / r)) * (h > 0)

    all_c = list(coords)
    all_z = list(Z)
    sim_vals = np.empty(m)
    order = rng.permutation(m)

    for idx in order:
        target = grid[idx]
        ac = np.array(all_c)
        az = np.array(all_z)
        dists = np.sqrt(((ac - target) ** 2).sum(-1))
        nn_idx = np.argsort(dists)[:max_neighbours]
        c_nn = ac[nn_idx]
        z_nn = az[nn_idx]
        ns = len(nn_idx)

        dist_nn = np.sqrt(((c_nn[:, None, :] - c_nn[None, :, :]) ** 2).sum(-1))
        G = _gamma(dist_nn)
        A = np.zeros((ns + 1, ns + 1))
        A[:ns, :ns] = G
        A[:ns, ns] = 1.0
        A[ns, :ns] = 1.0

        d0 = np.sqrt(((c_nn - target) ** 2).sum(-1))
        g0 = _gamma(d0)
        b = np.zeros(ns + 1)
        b[:ns] = g0
        b[ns] = 1.0

        try:
            lam = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            lam = np.zeros(ns + 1)
            lam[:ns] = 1.0 / ns

        w = lam[:ns]
        pred = w @ z_nn
        var = max(w @ g0 + lam[ns], 1e-10)
        sim_vals[idx] = rng.normal(pred, np.sqrt(var))

        all_c.append(target)
        all_z.append(sim_vals[idx])

    return SpatialResult(
        name="sequential_gaussian_sim",
        statistic=float(np.mean(sim_vals)),
        p_value=None,
        extra={"simulated_values": sim_vals, "grid": grid},
    )


sgsqs = sequential_gaussian_sim


def cheatsheet() -> str:
    return "sequential_gaussian_sim({}) -> Sequential Gaussian simulation."

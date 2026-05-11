"""Turning bands simulation."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def turning_bands_sim(
    coords: np.ndarray,
    cov_model: str = "exponential",
    cov_params: dict | None = None,
    n_bands: int = 100,
    seed: int = 42,
) -> SpatialResult:
    r"""Simulate a GRF using the turning bands method.

    Projects 2D coordinates onto random 1D lines, simulates 1D
    processes, and averages over bands.

    Parameters
    ----------
    coords : np.ndarray
        Simulation coordinates, shape ``(n, 2)``.
    cov_model : str
        ``"exponential"`` or ``"gaussian"``.
    cov_params : dict, optional
        ``{"sill", "range", "nugget"}``.
    n_bands : int
        Number of random lines (bands).
    seed : int
        RNG seed.

    Returns
    -------
    SpatialResult
        ``statistic`` is mean of simulated field.
        ``extra`` has ``simulated_values``, ``n_bands``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 7.

    .. epigraph::

        "I want to be your canary." -- FF9
    """
    rng = np.random.default_rng(seed)
    coords = np.asarray(coords, dtype=np.float64)
    n = len(coords)
    params = cov_params or {"sill": 1.0, "range": 1.0, "nugget": 0.0}
    sill = params.get("sill", 1.0)
    r = params.get("range", 1.0)

    Z = np.zeros(n)
    for _ in range(n_bands):
        theta = rng.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(theta), np.sin(theta)])
        proj = coords @ direction

        sorted_idx = np.argsort(proj)
        proj_sorted = proj[sorted_idx]
        diffs = np.diff(proj_sorted, prepend=proj_sorted[0] - 1.0)
        diffs = np.maximum(np.abs(diffs), 1e-10)

        if cov_model == "gaussian":
            rho = np.exp(-((diffs / r) ** 2))
        else:
            rho = np.exp(-diffs / r)

        y = np.empty(n)
        y[0] = rng.standard_normal()
        for j in range(1, n):
            y[j] = rho[j] * y[j - 1] + np.sqrt(max(1 - rho[j] ** 2, 0)) * rng.standard_normal()

        unsorted = np.empty(n)
        unsorted[sorted_idx] = y
        Z += unsorted

    Z *= np.sqrt(sill) / np.sqrt(n_bands)

    return SpatialResult(
        name="turning_bands_sim",
        statistic=float(np.mean(Z)),
        p_value=None,
        extra={"simulated_values": Z, "n_bands": n_bands},
    )


sgtbn = turning_bands_sim


def cheatsheet() -> str:
    return "turning_bands_sim({}) -> Turning bands simulation."

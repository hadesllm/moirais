"""Simulated annealing for spatial pattern optimization."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def simulated_annealing_spatial(
    Z: np.ndarray,
    coords: np.ndarray,
    target_variogram: callable | None = None,
    n_iter: int = 1000,
    T0: float = 1.0,
    cooling: float = 0.95,
    seed: int = 42,
) -> SpatialResult:
    r"""Optimize a spatial field to match a target variogram.

    Uses simulated annealing with random value swaps.

    Parameters
    ----------
    Z : np.ndarray
        Initial field values, shape ``(n,)``.
    coords : np.ndarray
        Observation coordinates, shape ``(n, 2)``.
    target_variogram : callable, optional
        ``f(h) -> gamma``. Defaults to exponential with range 1.
    n_iter : int
        Number of iterations.
    T0 : float
        Initial temperature.
    cooling : float
        Cooling factor.
    seed : int
        RNG seed.

    Returns
    -------
    SpatialResult
        ``statistic`` is final objective value.
        ``extra`` has ``optimized_field``, ``energy_history``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 7.

    .. epigraph::

    """
    rng = np.random.default_rng(seed)
    Z = np.asarray(Z, dtype=np.float64).ravel().copy()
    coords = np.asarray(coords, dtype=np.float64)
    n = len(Z)

    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    if target_variogram is None:

        def target_variogram(h):
            return 1.0 * (1.0 - np.exp(-h / 1.0)) * (h > 0)

    def _energy(z):
        bins = np.linspace(0, np.max(dist) * 0.5, 10)
        energy = 0.0
        for k in range(len(bins) - 1):
            mask = (dist > bins[k]) & (dist <= bins[k + 1])
            if mask.sum() == 0:
                continue
            diff = z[:, None] - z[None, :]
            emp_gamma = 0.5 * np.mean(diff[mask] ** 2)
            h_mid = 0.5 * (bins[k] + bins[k + 1])
            target = target_variogram(h_mid)
            energy += (emp_gamma - target) ** 2
        return energy

    best_Z = Z.copy()
    best_E = _energy(Z)
    curr_E = best_E
    T = T0
    history = [curr_E]

    for _ in range(n_iter):
        i, j = rng.choice(n, 2, replace=False)
        Z[i], Z[j] = Z[j], Z[i]
        new_E = _energy(Z)
        dE = new_E - curr_E
        if dE < 0 or rng.random() < np.exp(-dE / max(T, 1e-10)):
            curr_E = new_E
            if curr_E < best_E:
                best_E = curr_E
                best_Z = Z.copy()
        else:
            Z[i], Z[j] = Z[j], Z[i]
        T *= cooling
        history.append(curr_E)

    return SpatialResult(
        name="simulated_annealing_spatial",
        statistic=float(best_E),
        p_value=None,
        extra={
            "optimized_field": best_Z,
            "energy_history": np.array(history),
        },
    )


sgsa = simulated_annealing_spatial


def cheatsheet() -> str:
    return "simulated_annealing_spatial({}) -> Simulated annealing for spatial pattern optimization."

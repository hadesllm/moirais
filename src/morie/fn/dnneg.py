# morie.fn — function file (hadesllm/morie)
"""Density of ideal points for negative-weight respondents."""

from __future__ import annotations

from ._containers import DescriptiveResult


def density_negative_weights(positions, weights, n_points: int = 100) -> DescriptiveResult:
    """Kernel density estimate for respondents with negative A-M weights.

    :param positions: Ideal point estimates.
    :param weights: Per-respondent weights (beta).
    :param n_points: Grid points for density.
    :return: DescriptiveResult with density data.

    .. epigraph:: "United States of Smash!" -- All Might, My Hero Academia
    """
    import numpy as np

    pos = np.asarray(positions, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    mask = w < 0
    pos_sub = pos[mask]
    if len(pos_sub) < 2:
        return DescriptiveResult(name="density_negative_weights", value=0, extra={"grid": [], "density": [], "n": 0})
    grid = np.linspace(pos_sub.min() - 1, pos_sub.max() + 1, n_points)
    bw = 1.06 * pos_sub.std() * len(pos_sub) ** (-0.2)
    if bw == 0:
        bw = 1.0
    density = np.zeros(n_points)
    for p in pos_sub:
        density += np.exp(-0.5 * ((grid - p) / bw) ** 2)
    density /= len(pos_sub) * bw * np.sqrt(2 * np.pi)
    return DescriptiveResult(
        name="density_negative_weights",
        value=int(mask.sum()),
        extra={"grid": grid.tolist(), "density": density.tolist(), "n": int(mask.sum()), "bandwidth": float(bw)},
    )


dnneg = density_negative_weights


def cheatsheet() -> str:
    return "density_negative_weights({}) -> Density of ideal points for negative-weight respondents."

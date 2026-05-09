# moirais.fn — function file (hadesllm/moirais)
"""Posterior density estimation via KDE."""

from __future__ import annotations

from ._containers import DescriptiveResult


def posterior_density_data(chain) -> DescriptiveResult:
    """KDE density estimate of posterior samples.

    .. epigraph:: "A Lannister always pays his debts." -- Tyrion, Game of Thrones
    """
    import numpy as np

    chain = np.asarray(chain, dtype=float).ravel()
    n = len(chain)
    bw = 1.06 * np.std(chain, ddof=1) * n ** (-0.2)
    grid = np.linspace(np.min(chain) - 3 * bw, np.max(chain) + 3 * bw, 200)
    density = np.zeros_like(grid)
    for x in chain:
        density += np.exp(-0.5 * ((grid - x) / bw) ** 2)
    density /= n * bw * np.sqrt(2 * np.pi)
    return DescriptiveResult(
        name="posterior_density_data",
        value=float(grid[np.argmax(density)]),
        extra={
            "grid": grid,
            "density": density,
            "mode": float(grid[np.argmax(density)]),
            "bandwidth": float(bw),
            "n": n,
        },
    )


pstdn = posterior_density_data


def cheatsheet() -> str:
    return "posterior_density_data({}) -> Posterior density estimation via KDE."

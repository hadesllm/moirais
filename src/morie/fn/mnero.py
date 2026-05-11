# morie.fn — function file (hadesllm/morie)
"""Study the past if you would define the future. — Confucius"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def cellular_automaton(
    grid: np.ndarray | None = None,
    *,
    size: int = 50,
    n_steps: int = 100,
    rule: str = "life",
    seed: int | None = 42,
    density: float = 0.3,
) -> DescriptiveResult:
    """Run a 2-D cellular automaton simulation.

    Supports Conway's Game of Life and a simple urban growth model where
    cells become "developed" based on neighbour density and a stochastic
    threshold.

    Parameters
    ----------
    grid : ndarray or None
        Initial state (binary 0/1). If None, generates randomly.
    size : int
        Grid side length (used when grid is None).
    n_steps : int
        Number of simulation steps.
    rule : str
        ``"life"`` for Conway's Game of Life or ``"urban"`` for urban growth.
    seed : int or None
        Random seed.
    density : float
        Initial density of live cells (when grid is None).

    Returns
    -------
    DescriptiveResult
        ``value`` is the final density of live cells; ``extra`` has density
        history and grid dimensions.
    """
    rng = np.random.default_rng(seed)
    if grid is None:
        grid = (rng.random((size, size)) < density).astype(int)
    else:
        grid = np.asarray(grid, dtype=int)
        if grid.ndim != 2:
            raise ValueError("grid must be 2-D")
    rows, cols = grid.shape

    def _count_neighbors(g):
        n = np.zeros_like(g)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                n += np.roll(np.roll(g, dr, axis=0), dc, axis=1)
        return n

    densities = [float(grid.mean())]
    for step in range(n_steps):
        nb = _count_neighbors(grid)
        if rule == "life":
            birth = (grid == 0) & (nb == 3)
            survive = (grid == 1) & ((nb == 2) | (nb == 3))
            grid = (birth | survive).astype(int)
        elif rule == "urban":
            prob = nb / 8.0
            develop = (grid == 0) & (rng.random(grid.shape) < prob * 0.5)
            decay = (grid == 1) & (rng.random(grid.shape) < 0.01)
            grid = grid.copy()
            grid[develop] = 1
            grid[decay] = 0
        else:
            raise ValueError(f"rule must be 'life' or 'urban', got '{rule}'")
        densities.append(float(grid.mean()))

    return DescriptiveResult(
        name=f"Cellular Automaton ({rule})",
        value=float(grid.mean()),
        extra={
            "density_history": densities[:: max(1, len(densities) // 20)],
            "final_live_cells": int(grid.sum()),
            "grid_shape": [rows, cols],
            "n_steps": n_steps,
            "rule": rule,
        },
    )


mnero = cellular_automaton


def cheatsheet() -> str:
    return "cellular_automaton({}) -> Cellular automata urban growth model. 'Welcome to the real w"

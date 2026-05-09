"""Voronoi cells on a flat torus."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def torus_voronoi(
    points: np.ndarray | None = None,
    n: int = 20,
    a: float = 1.0,
    b: float = 1.0,
) -> DescriptiveResult:
    """Compute nearest-neighbour Voronoi assignment on a flat torus [0,a)x[0,b).

    Uses toroidal (wrapped) distance: for each query grid point, find
    the nearest seed under periodic boundary conditions.

    :param points: Seed points (n_seeds, 2) in [0,a) x [0,b). Random if None.
    :param n: Grid resolution per side for assignment.
    :param a: Torus width.
    :param b: Torus height.
    :return: DescriptiveResult with assignment grid and cell areas.
    """
    if a <= 0 or b <= 0:
        raise ValueError(f"Dimensions must be positive, got a={a}, b={b}.")
    rng = np.random.default_rng(42)
    if points is None:
        points = rng.uniform(0, 1, (10, 2)) * np.array([a, b])
    points = np.asarray(points, dtype=float)
    xs = np.linspace(0, a, n, endpoint=False)
    ys = np.linspace(0, b, n, endpoint=False)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    dx = grid[:, 0:1] - points[:, 0]
    dy = grid[:, 1:2] - points[:, 1]
    dx = np.minimum(np.abs(dx), a - np.abs(dx))
    dy = np.minimum(np.abs(dy), b - np.abs(dy))
    dist2 = dx**2 + dy**2
    assignment = np.argmin(dist2, axis=1).reshape(n, n)
    n_seeds = len(points)
    cell_counts = np.array([np.sum(assignment == i) for i in range(n_seeds)])
    cell_areas = cell_counts * (a * b) / (n * n)
    return DescriptiveResult(
        name="torus_voronoi",
        value=float(n_seeds),
        extra={
            "assignment": assignment,
            "seeds": points,
            "cell_areas": cell_areas,
            "n_seeds": n_seeds,
        },
    )


def cheatsheet() -> str:
    return "torus_voronoi(points, n, a, b) -> Voronoi cells on flat torus"

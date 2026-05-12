# morie.fn -- function file (hadesllm/morie)
"""Waste no more time arguing what a good person should be. Be one. -- Marcus Aurelius"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def rrt_plan(
    *,
    start: tuple[float, float] = (0.0, 0.0),
    goal: tuple[float, float] = (10.0, 10.0),
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 12.0, 12.0),
    obstacles: list[tuple[float, float, float]] | None = None,
    step_size: float = 0.5,
    max_iter: int = 2000,
    goal_radius: float = 0.5,
    seed: int | None = 42,
) -> DescriptiveResult:
    """Rapidly-exploring Random Tree (RRT) path planner in 2-D.

    Builds a tree from *start* toward *goal* while avoiding circular
    obstacles.

    Parameters
    ----------
    start, goal : tuple
        (x, y) coordinates.
    bounds : tuple
        (x_min, y_min, x_max, y_max) workspace limits.
    obstacles : list of (cx, cy, r) or None
        Circular obstacles (centre x, centre y, radius).
    step_size : float
        Maximum extension distance per iteration.
    max_iter : int
        Maximum tree growth iterations.
    goal_radius : float
        Distance within which the goal is considered reached.
    seed : int or None
        Random seed.

    Returns
    -------
    DescriptiveResult
        ``value`` is the path length (inf if not found); ``extra`` has
        the path waypoints and tree size.
    """
    rng = np.random.default_rng(seed)
    if obstacles is None:
        obstacles = []
    x_min, y_min, x_max, y_max = bounds

    nodes = [np.array(start)]
    parents = [-1]

    def _collides(p1, p2):
        for cx, cy, r in obstacles:
            c = np.array([cx, cy])
            d = p2 - p1
            f = p1 - c
            a = np.dot(d, d)
            b = 2 * np.dot(f, d)
            cc = np.dot(f, f) - r * r
            disc = b * b - 4 * a * cc
            if disc >= 0 and a > 0:
                disc_sqrt = np.sqrt(disc)
                t1 = (-b - disc_sqrt) / (2 * a)
                t2 = (-b + disc_sqrt) / (2 * a)
                if t1 <= 1 and t2 >= 0:
                    return True
        return False

    found = False
    for _ in range(max_iter):
        if rng.random() < 0.1:
            sample = np.array(goal)
        else:
            sample = np.array(
                [
                    rng.uniform(x_min, x_max),
                    rng.uniform(y_min, y_max),
                ]
            )

        dists = [np.linalg.norm(sample - n) for n in nodes]
        nearest_idx = int(np.argmin(dists))
        nearest = nodes[nearest_idx]

        direction = sample - nearest
        dist = np.linalg.norm(direction)
        if dist < 1e-10:
            continue
        direction = direction / dist
        new_pt = nearest + direction * min(step_size, dist)

        if not (x_min <= new_pt[0] <= x_max and y_min <= new_pt[1] <= y_max):
            continue
        if _collides(nearest, new_pt):
            continue

        nodes.append(new_pt)
        parents.append(nearest_idx)

        if np.linalg.norm(new_pt - np.array(goal)) < goal_radius:
            found = True
            break

    if not found:
        return DescriptiveResult(
            name="RRT Planner",
            value=float("inf"),
            extra={"path": [], "tree_size": len(nodes), "found": False},
        )

    path = [nodes[-1].tolist()]
    idx = len(nodes) - 1
    while parents[idx] != -1:
        idx = parents[idx]
        path.append(nodes[idx].tolist())
    path.reverse()

    length = sum(np.linalg.norm(np.array(path[i + 1]) - np.array(path[i])) for i in range(len(path) - 1))

    return DescriptiveResult(
        name="RRT Planner",
        value=float(length),
        extra={
            "path": path,
            "waypoints": len(path),
            "tree_size": len(nodes),
            "found": True,
        },
    )


nebzr = rrt_plan


def cheatsheet() -> str:
    return "Waste no more time arguing what a good person should be. Be one. -- Marcus Aurelius"

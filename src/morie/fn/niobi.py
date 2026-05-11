# morie.fn — function file (hadesllm/morie)
"""A* pathfinding algorithm. 'Lock and load.' -- Niobe"""

from __future__ import annotations

import heapq

import numpy as np

from ._containers import DescriptiveResult


def astar_path(
    grid: np.ndarray,
    *,
    start: tuple[int, int] = (0, 0),
    goal: tuple[int, int] | None = None,
    diagonal: bool = True,
) -> DescriptiveResult:
    """A* shortest path on a 2-D grid.

    Obstacles are cells with value <= 0. Traversable cells have positive cost
    (1 = normal, higher = more expensive). Uses Euclidean heuristic when
    diagonal movement is allowed, Manhattan otherwise.

    Parameters
    ----------
    grid : ndarray
        2-D cost grid (rows x cols). Values <= 0 are impassable.
    start : tuple
        (row, col) start position.
    goal : tuple or None
        (row, col) goal position. Defaults to bottom-right corner.
    diagonal : bool
        Allow 8-connected movement (True) or 4-connected (False).

    Returns
    -------
    DescriptiveResult
        ``value`` is the total path cost; ``extra`` has the path as a list
        of (row, col) tuples and the number of nodes explored.
    """
    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim != 2:
        raise ValueError("grid must be 2-D")
    rows, cols = grid.shape
    if goal is None:
        goal = (rows - 1, cols - 1)

    sr, sc = start
    gr, gc = goal
    if not (0 <= sr < rows and 0 <= sc < cols):
        raise ValueError(f"start {start} out of bounds")
    if not (0 <= gr < rows and 0 <= gc < cols):
        raise ValueError(f"goal {goal} out of bounds")
    if grid[sr, sc] <= 0:
        raise ValueError("start cell is impassable")
    if grid[gr, gc] <= 0:
        raise ValueError("goal cell is impassable")

    if diagonal:
        moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        def heuristic(r, c):
            return np.sqrt((r - gr) ** 2 + (c - gc) ** 2)
    else:
        moves = [(-1, 0), (0, -1), (0, 1), (1, 0)]

        def heuristic(r, c):
            return abs(r - gr) + abs(c - gc)

    open_set = [(heuristic(sr, sc), 0.0, sr, sc)]
    g_score = {(sr, sc): 0.0}
    came_from = {}
    explored = 0

    while open_set:
        _f, g, r, c = heapq.heappop(open_set)
        explored += 1
        if (r, c) == goal:
            path = [(r, c)]
            while (r, c) in came_from:
                r, c = came_from[(r, c)]
                path.append((r, c))
            path.reverse()
            return DescriptiveResult(
                name="A* Pathfinding",
                value=float(g),
                extra={
                    "path": path,
                    "path_length": len(path),
                    "nodes_explored": explored,
                    "grid_shape": list(grid.shape),
                },
            )

        if g > g_score.get((r, c), float("inf")):
            continue

        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] > 0:
                step = np.sqrt(dr**2 + dc**2) * grid[nr, nc]
                ng = g + step
                if ng < g_score.get((nr, nc), float("inf")):
                    g_score[(nr, nc)] = ng
                    came_from[(nr, nc)] = (r, c)
                    heapq.heappush(open_set, (ng + heuristic(nr, nc), ng, nr, nc))

    return DescriptiveResult(
        name="A* Pathfinding",
        value=float("inf"),
        extra={
            "path": [],
            "path_length": 0,
            "nodes_explored": explored,
            "reachable": False,
        },
    )


niobi = astar_path


def cheatsheet() -> str:
    return "astar_path({}) -> A* pathfinding algorithm. 'Lock and load.' -- Niobe"

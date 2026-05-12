# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Arrow plot / directed graph layout. 'You have failed this city!' -- Green Arrow"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def directed_layout(
    adjacency: np.ndarray | list[list[float]],
    *,
    method: str = "force",
    n_iter: int = 100,
    seed: int | None = None,
) -> DescriptiveResult:
    """Compute 2-D layout positions for a directed graph via force-directed placement.

    Uses Fruchterman-Reingold-style repulsion/attraction forces on the
    adjacency matrix to produce readable node positions.

    Parameters
    ----------
    adjacency : array (n, n)
        Adjacency or weight matrix (non-zero = edge).
    method : str
        ``'Our greatest glory is not in never falling, but in rising every time we fall. -- Confucius'`` (Fruchterman-Reingold) or ``'circular'``.
    n_iter : int
        Iterations for force-directed layout.
    seed : int or None
        Random seed for initial positions.

    Returns
    -------
    DescriptiveResult
        ``value`` = dict with ``'positions'`` (n x 2 list) and ``'edges'``.
    """
    A = np.asarray(adjacency, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    n = A.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 nodes")
    rng = np.random.default_rng(seed)
    if method == "circular":
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos = np.column_stack([np.cos(angles), np.sin(angles)])
    elif method == "force":
        pos = rng.uniform(-1, 1, (n, 2))
        area = 4.0
        k = np.sqrt(area / n)
        temp = 1.0
        for _ in range(n_iter):
            disp = np.zeros_like(pos)
            for i in range(n):
                delta = pos[i] - pos
                dist = np.linalg.norm(delta, axis=1)
                dist = np.maximum(dist, 0.01)
                rep = (k**2 / dist)[:, None] * delta / dist[:, None]
                rep[i] = 0
                disp[i] += rep.sum(axis=0)
            for i in range(n):
                for j in range(n):
                    if A[i, j] != 0 or A[j, i] != 0:
                        delta = pos[j] - pos[i]
                        d = max(np.linalg.norm(delta), 0.01)
                        attr = (d / k) * delta / d
                        disp[i] += attr
                        disp[j] -= attr
            norms = np.linalg.norm(disp, axis=1, keepdims=True)
            norms = np.maximum(norms, 0.01)
            pos += (disp / norms) * np.minimum(temp, np.linalg.norm(disp, axis=1, keepdims=True))
            temp *= 0.95
    else:
        raise ValueError(f"Unknown method: {method}")
    edges = []
    for i in range(n):
        for j in range(n):
            if A[i, j] != 0:
                edges.append({"from": i, "to": j, "weight": float(A[i, j])})
    return DescriptiveResult(
        name=f"Directed graph layout ({method})",
        value={"positions": pos.tolist(), "edges": edges},
        extra={"n_nodes": n, "n_edges": len(edges), "method": method},
    )


arrow = directed_layout


def cheatsheet() -> str:
    return "directed_layout({}) -> Arrow plot / directed graph layout. 'You have failed this ci"

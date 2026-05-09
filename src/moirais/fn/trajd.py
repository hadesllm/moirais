"""Trajectory distance via discrete Frechet distance."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def trajectory_distance(
    path_a: np.ndarray,
    path_b: np.ndarray,
    metric: str = "frechet",
) -> SpatialResult:
    r"""Compute distance between two movement trajectories.

    Implements the discrete Frechet distance:

    .. math::

        \delta_F(P, Q) = \min_{\sigma, \tau}
        \max_{k} \| P_{\sigma(k)} - Q_{\tau(k)} \|

    computed via dynamic programming over the coupling sequence.

    For ``metric="hausdorff"``, returns the directed Hausdorff distance:

    .. math::

        d_H(P, Q) = \max\!\bigl(
        \max_i \min_j \|p_i - q_j\|,\;
        \max_j \min_i \|p_i - q_j\|\bigr)

    Parameters
    ----------
    path_a : np.ndarray
        First trajectory, shape ``(m, d)`` where *d* is 2 or 3.
    path_b : np.ndarray
        Second trajectory, shape ``(k, d)``.
    metric : str
        ``"frechet"`` (default) or ``"hausdorff"``.

    Returns
    -------
    SpatialResult
        ``statistic`` is the distance value.
        ``extra`` contains ``metric`` and ``coupling`` (Frechet only).

    References
    ----------
    Eiter T, Mannila H (1994). Computing discrete Frechet distance.
    Technical Report CD-TR 94/64, TU Vienna.

    Alt H, Godau M (1995). Computing the Frechet distance between two
    polygonal curves. *International Journal of Computational Geometry
    & Applications*, 5(1-2), 75-91.

    Buchin K, Buchin M, Gudmundsson J (2010). Detecting single file
    movement. *Proceedings of 18th ACM SIGSPATIAL*, 288-297.
    """
    P = np.asarray(path_a, dtype=np.float64)
    Q = np.asarray(path_b, dtype=np.float64)

    if P.ndim != 2 or Q.ndim != 2:
        raise ValueError("paths must be 2-D arrays")
    if P.shape[1] != Q.shape[1]:
        raise ValueError("paths must have same dimensionality")

    m, k = P.shape[0], Q.shape[0]

    if metric == "hausdorff":
        dists = np.sqrt(((P[:, None, :] - Q[None, :, :]) ** 2).sum(axis=2))
        d_fwd = np.max(np.min(dists, axis=1))
        d_bwd = np.max(np.min(dists, axis=0))
        return SpatialResult(
            name="Hausdorff_Distance",
            statistic=float(max(d_fwd, d_bwd)),
            p_value=None,
            extra={"metric": "hausdorff"},
        )

    ca = np.full((m, k), -1.0)

    def _dist(i, j):
        return float(np.linalg.norm(P[i] - Q[j]))

    ca[0, 0] = _dist(0, 0)
    for i in range(1, m):
        ca[i, 0] = max(ca[i - 1, 0], _dist(i, 0))
    for j in range(1, k):
        ca[0, j] = max(ca[0, j - 1], _dist(0, j))

    for i in range(1, m):
        for j in range(1, k):
            ca[i, j] = max(
                min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1]),
                _dist(i, j),
            )

    i, j = m - 1, k - 1
    coupling = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            prev = np.argmin([ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1]])
            if prev == 0:
                i -= 1
            elif prev == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
        coupling.append((i, j))
    coupling.reverse()

    return SpatialResult(
        name="Frechet_Distance",
        statistic=float(ca[m - 1, k - 1]),
        p_value=None,
        extra={"metric": "frechet", "coupling": coupling},
    )


def cheatsheet() -> str:
    return "trajectory_distance({}) -> Trajectory distance via discrete Frechet distance."

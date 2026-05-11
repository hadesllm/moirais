# morie.fn — function file (hadesllm/morie)
"""Edge-corrected Ripley's K with L-function transformation."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def ripley_k_corrected(
    points: np.ndarray,
    bbox: tuple[float, float, float, float] | None = None,
    distances: np.ndarray | None = None,
    n_distances: int = 25,
    correction: str = "ripley",
) -> SpatialResult:
    r"""Edge-corrected Ripley's K-function with variance-stabilizing L.

    Computes the isotropic edge-corrected estimator:

    .. math::

        \hat{K}(d) = \frac{|A|}{n^2}
        \sum_{i \neq j} \frac{\mathbf{1}(\|s_i - s_j\| \le d)}{e_{ij}}

    where :math:`e_{ij}` is the proportion of the circle centered at
    :math:`s_i` passing through :math:`s_j` that lies within the study
    region (Ripley's isotropic correction).

    The variance-stabilizing L-function (Besag 1977):

    .. math::

        \hat{L}(d) = \sqrt{\frac{\hat{K}(d)}{\pi}} - d

    Parameters
    ----------
    points : np.ndarray
        Point coordinates, shape ``(n, 2)``.
    bbox : tuple, optional
        Bounding box ``(xmin, ymin, xmax, ymax)``. Default: data extent.
    distances : np.ndarray, optional
        Evaluation distances. Default: ``n_distances`` equispaced values.
    n_distances : int
        Number of distances if not provided. Default 25.
    correction : str
        Edge correction: ``"ripley"`` (isotropic) or ``"none"``.

    Returns
    -------
    SpatialResult
        ``statistic`` is max absolute L(d) (departure from CSR).
        ``extra`` contains ``K``, ``L``, ``distances``, ``area``.

    References
    ----------
    Ripley BD (1976). The second-order analysis of stationary point
    processes. *Journal of Applied Probability*, 13(2), 255-266.

    Besag JE (1977). Discussion on Dr. Ripley's paper. *Journal of the
    Royal Statistical Society B*, 39(2), 193-195.

    Diggle PJ (2003). *Statistical Analysis of Spatial Point Patterns*,
    2nd ed. Arnold, London.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be (n, 2)")
    n = pts.shape[0]

    if bbox is None:
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)
    else:
        xmin, ymin, xmax, ymax = bbox

    area = (xmax - xmin) * (ymax - ymin)
    if area <= 0:
        raise ValueError("bounding box must have positive area")

    max_d = min(xmax - xmin, ymax - ymin) / 2.0
    if distances is None:
        distances = np.linspace(0, max_d, n_distances + 1)[1:]
    else:
        distances = np.asarray(distances, dtype=np.float64)

    dmat = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))

    K = np.empty(len(distances))

    for di, d in enumerate(distances):
        indicator = (dmat <= d).astype(np.float64)
        np.fill_diagonal(indicator, 0.0)

        if correction == "ripley":
            weights = np.ones_like(indicator)
            for i in range(n):
                for j in range(n):
                    if i == j or dmat[i, j] > d:
                        continue
                    r = dmat[i, j]
                    dx_min = min(pts[i, 0] - xmin, xmax - pts[i, 0])
                    dy_min = min(pts[i, 1] - ymin, ymax - pts[i, 1])
                    d_edge = min(dx_min, dy_min)
                    if r <= d_edge:
                        weights[i, j] = 1.0
                    else:
                        theta = np.arccos(min(d_edge / r, 1.0))
                        weights[i, j] = 1.0 / (1.0 - theta / np.pi)
            K[di] = area / (n * (n - 1)) * np.sum(indicator * weights)
        else:
            K[di] = area / (n * (n - 1)) * np.sum(indicator)

    L = np.sqrt(K / np.pi) - distances

    return SpatialResult(
        name="Ripley_K_Corrected",
        statistic=float(np.max(np.abs(L))),
        p_value=None,
        extra={
            "K": K,
            "L": L,
            "distances": distances,
            "area": area,
        },
    )


def cheatsheet() -> str:
    return "ripley_k_corrected({}) -> Edge-corrected Ripley's K with L-function transformation."

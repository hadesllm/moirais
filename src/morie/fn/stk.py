"""Space-time K function."""

import numpy as np

from ._containers import DescriptiveResult


def space_time_k(
    points: np.ndarray,
    times: np.ndarray,
    spatial_dists: np.ndarray | None = None,
    temporal_dists: np.ndarray | None = None,
    n_s: int = 10,
    n_t: int = 10,
) -> DescriptiveResult:
    """
    Compute the space-time K function.

    Extends Ripley's K to the spatiotemporal domain:

    .. math::

        \\hat{K}_{ST}(s,t) = \\frac{|A| \\cdot T}{n^2}
        \\sum_{i \\neq j} \\mathbf{1}(d_{ij} \\le s, |t_i - t_j| \\le t)

    :param points: (n, 2) spatial coordinates.
    :param times: (n,) temporal values.
    :param spatial_dists: Distances to evaluate (optional).
    :param temporal_dists: Temporal lags to evaluate (optional).
    :param n_s: Number of spatial distances.
    :param n_t: Number of temporal distances.
    :return: DescriptiveResult with K_st matrix in extra.

    References
    ----------
    Diggle PJ et al. (1995). Second-order analysis of space-time
    point processes. Statistical Methods in Medical Research, 4, 124-136.
    """
    pts = np.asarray(points, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64).ravel()
    n = pts.shape[0]
    x_rng = pts[:, 0].max() - pts[:, 0].min()
    y_rng = pts[:, 1].max() - pts[:, 1].min()
    area = x_rng * y_rng if x_rng > 0 and y_rng > 0 else 1.0
    t_rng = t.max() - t.min() if t.max() > t.min() else 1.0
    dmat = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
    tmat = np.abs(t[:, None] - t[None, :])
    if spatial_dists is None:
        spatial_dists = np.linspace(0, dmat.max() / 2, n_s)
    if temporal_dists is None:
        temporal_dists = np.linspace(0, tmat.max() / 2, n_t)
    spatial_dists = np.asarray(spatial_dists)
    temporal_dists = np.asarray(temporal_dists)
    K_st = np.zeros((len(spatial_dists), len(temporal_dists)))
    for si, s in enumerate(spatial_dists):
        for ti, tv in enumerate(temporal_dists):
            count = np.sum((dmat <= s) & (tmat <= tv)) - n
            K_st[si, ti] = area * t_rng / (n * n) * count
    return DescriptiveResult(
        name="space_time_k",
        value=float(K_st.sum()),
        extra={"K_st": K_st, "spatial_dists": spatial_dists, "temporal_dists": temporal_dists, "n": n},
    )


stk = space_time_k


def cheatsheet() -> str:
    return "space_time_k({}) -> Space-time K function."

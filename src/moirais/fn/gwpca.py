# moirais.fn — function file (hadesllm/moirais)
"""Geographically weighted PCA (Harris et al. 2011)."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def gw_pca(
    X: np.ndarray,
    coords: np.ndarray,
    bandwidth: float | None = None,
    n_components: int = 2,
    kernel: str = "gaussian",
) -> SpatialResult:
    r"""Geographically weighted principal component analysis.

    At each location *i*, computes a locally weighted covariance
    matrix and extracts principal components:

    .. math::

        \Sigma_i = \frac{1}{\sum_j w_{ij}}
        \sum_{j=1}^n w_{ij} (x_j - \bar{x}_i)(x_j - \bar{x}_i)^\top

    where :math:`w_{ij} = K(d_{ij}/h)` is a spatial kernel weight and
    :math:`\bar{x}_i = \sum_j w_{ij} x_j / \sum_j w_{ij}`.

    The local eigenvalues reveal how the importance of principal
    components varies across space.

    Parameters
    ----------
    X : np.ndarray
        Data matrix, shape ``(n, p)``.
    coords : np.ndarray
        Spatial coordinates, shape ``(n, 2)``.
    bandwidth : float, optional
        Kernel bandwidth. Default: median pairwise distance.
    n_components : int
        Number of PCs to retain. Default 2.
    kernel : str
        ``"gaussian"`` or ``"bisquare"``. Default ``"gaussian"``.

    Returns
    -------
    SpatialResult
        ``statistic`` is the mean proportion of variance explained by
        the first component across locations.
        ``extra`` contains ``local_eigenvalues`` (n x n_comp),
        ``local_loadings`` (n x p x n_comp),
        ``local_scores`` (n x n_comp).

    References
    ----------
    Harris P, Brunsdon C, Charlton M (2011). Geographically weighted
    principal components analysis. *International Journal of
    Geographical Information Science*, 25(10), 1717-1736.

    Lloyd CD (2010). Exploring population density and advantage/
    disadvantage at the neighbourhood level. *Applied Spatial Analysis
    and Policy*, 3, 127-148.

    Fotheringham AS, Brunsdon C, Charlton ME (2002). *Geographically
    Weighted Regression*. Wiley.
    """
    Xm = np.asarray(X, dtype=np.float64)
    xy = np.asarray(coords, dtype=np.float64)
    n = Xm.shape[0]

    if Xm.ndim == 1:
        Xm = Xm.reshape(-1, 1)
    p = Xm.shape[1]
    n_components = min(n_components, p)

    if xy.shape != (n, 2):
        raise ValueError("coords must be (n, 2)")

    dmat = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2))

    if bandwidth is None:
        upper = dmat[np.triu_indices(n, k=1)]
        bandwidth = float(np.median(upper)) if len(upper) > 0 else 1.0
    bandwidth = max(bandwidth, 1e-10)

    def _kern(d):
        u = d / bandwidth
        if kernel == "bisquare":
            return np.where(u < 1, (1 - u**2) ** 2, 0.0)
        return np.exp(-0.5 * u**2)

    local_eig = np.empty((n, n_components))
    local_loadings = np.empty((n, p, n_components))
    local_scores = np.empty((n, n_components))
    prop_var_1 = np.empty(n)

    for i in range(n):
        w = _kern(dmat[i])
        w_sum = w.sum()
        if w_sum <= 0:
            local_eig[i] = 0.0
            local_loadings[i] = 0.0
            local_scores[i] = 0.0
            prop_var_1[i] = 0.0
            continue

        x_bar = (w[:, None] * Xm).sum(axis=0) / w_sum
        Xc = Xm - x_bar[None, :]
        Xw = Xc * np.sqrt(w[:, None])
        cov = (Xw.T @ Xw) / w_sum

        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        local_eig[i] = eigvals[:n_components]
        local_loadings[i] = eigvecs[:, :n_components]
        local_scores[i] = (Xm[i] - x_bar) @ eigvecs[:, :n_components]

        total_var = eigvals.sum()
        prop_var_1[i] = eigvals[0] / total_var if total_var > 0 else 0.0

    return SpatialResult(
        name="GW_PCA",
        statistic=float(np.mean(prop_var_1)),
        p_value=None,
        extra={
            "local_eigenvalues": local_eig,
            "local_loadings": local_loadings,
            "local_scores": local_scores,
            "prop_variance_pc1": prop_var_1,
            "bandwidth": bandwidth,
            "n_components": n_components,
        },
    )


def cheatsheet() -> str:
    return "gw_pca({}) -> Geographically weighted PCA (Harris et al. 2011)."

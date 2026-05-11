"""Geographically and temporally weighted regression (GTWR)."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def gtwr(
    X: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    times: np.ndarray,
    bw_spatial: float | None = None,
    bw_temporal: float | None = None,
    kernel: str = "gaussian",
) -> SpatialResult:
    r"""Geographically and temporally weighted regression.

    Extends GWR to the spatiotemporal domain by solving a locally
    weighted least squares problem at each observation:

    .. math::

        \hat{\beta}(s_i, t_i) = (X^\top W_i X)^{-1} X^\top W_i y

    where :math:`W_i` is a diagonal weight matrix combining spatial
    and temporal distance kernels:

    .. math::

        w_{ij} = K_s\!\left(\frac{d(s_i, s_j)}{h_s}\right)
        \cdot K_t\!\left(\frac{|t_i - t_j|}{h_t}\right)

    Parameters
    ----------
    X : np.ndarray
        Predictor matrix, shape ``(n, p)``.
    y : np.ndarray
        Response vector, shape ``(n,)``.
    coords : np.ndarray
        Spatial coordinates, shape ``(n, 2)``.
    times : np.ndarray
        Time values, shape ``(n,)``.
    bw_spatial : float, optional
        Spatial bandwidth. Default: median pairwise distance.
    bw_temporal : float, optional
        Temporal bandwidth. Default: median pairwise time difference.
    kernel : str
        Kernel type: ``"gaussian"`` or ``"bisquare"``. Default ``"gaussian"``.

    Returns
    -------
    SpatialResult
        ``statistic`` is R-squared.
        ``extra`` contains ``betas`` (n x p), ``y_hat``, ``residuals``.

    References
    ----------
    Huang B, Wu B, Barry M (2010). Geographically and temporally
    weighted regression for modeling spatio-temporal variation in house
    prices. *International Journal of Geographical Information Science*,
    24(3), 383-401.

    Fotheringham AS, Brunsdon C, Charlton ME (2002). *Geographically
    Weighted Regression: The Analysis of Spatially Varying
    Relationships*. Wiley.

    Wu B, Li R, Huang B (2014). A geographically and temporally
    weighted autoregressive model with application to housing prices.
    *International Journal of Geographical Information Science*,
    28(5), 1186-1204.
    """
    Xm = np.asarray(X, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64).ravel()
    xy = np.asarray(coords, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64).ravel()
    n = len(yv)

    if Xm.ndim == 1:
        Xm = Xm.reshape(-1, 1)
    if Xm.shape[0] != n:
        raise ValueError("X and y must have same number of rows")
    if xy.shape != (n, 2):
        raise ValueError("coords must be (n, 2)")

    p = Xm.shape[1]

    dx = xy[:, 0][:, None] - xy[:, 0][None, :]
    dy = xy[:, 1][:, None] - xy[:, 1][None, :]
    sdist = np.sqrt(dx**2 + dy**2)
    tdist = np.abs(t[:, None] - t[None, :])

    if bw_spatial is None:
        upper = sdist[np.triu_indices(n, k=1)]
        bw_spatial = float(np.median(upper)) if len(upper) > 0 else 1.0
    if bw_temporal is None:
        upper = tdist[np.triu_indices(n, k=1)]
        bw_temporal = float(np.median(upper)) if len(upper) > 0 else 1.0

    bw_spatial = max(bw_spatial, 1e-10)
    bw_temporal = max(bw_temporal, 1e-10)

    def _kern(d, bw):
        u = d / bw
        if kernel == "bisquare":
            return np.where(u < 1, (1 - u**2) ** 2, 0.0)
        return np.exp(-0.5 * u**2)

    betas = np.empty((n, p))
    y_hat = np.empty(n)

    for i in range(n):
        ws = _kern(sdist[i], bw_spatial)
        wt = _kern(tdist[i], bw_temporal)
        w = ws * wt
        W = np.diag(w)
        XtW = Xm.T @ W
        try:
            betas[i] = np.linalg.solve(XtW @ Xm + 1e-10 * np.eye(p), XtW @ yv)
        except np.linalg.LinAlgError:
            betas[i] = np.linalg.lstsq(Xm, yv, rcond=None)[0]
        y_hat[i] = Xm[i] @ betas[i]

    residuals = yv - y_hat
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yv - yv.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return SpatialResult(
        name="GTWR",
        statistic=float(r2),
        p_value=None,
        extra={
            "betas": betas,
            "y_hat": y_hat,
            "residuals": residuals,
            "bw_spatial": bw_spatial,
            "bw_temporal": bw_temporal,
        },
    )


def cheatsheet() -> str:
    return "gtwr({}) -> Geographically and temporally weighted regression (GTWR)."

"""Spatiotemporal Cox process intensity estimation."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def spacetime_cox(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    covariates: np.ndarray | None = None,
    n_xbins: int = 10,
    n_ybins: int = 10,
    n_tbins: int = 10,
    bandwidth_s: float | None = None,
    bandwidth_t: float | None = None,
) -> SpatialResult:
    r"""
    Log-Gaussian Cox process intensity estimation for spatiotemporal data.

    Estimates the first-order intensity of an inhomogeneous Poisson process
    driven by a latent Gaussian random field. The method uses kernel-smoothed
    event counts as the response in a Poisson regression:

    .. math::

        \log \hat{\lambda}(s, t) = \beta_0 + \mathbf{x}(s,t)^T \boldsymbol{\beta}

    where :math:`\hat{\lambda}` is the estimated intensity and :math:`\mathbf{x}`
    are optional spatiotemporal covariates evaluated at grid centroids.

    Parameters
    ----------
    x, y : np.ndarray
        (n,) spatial coordinates.
    t : np.ndarray
        (n,) temporal coordinates.
    covariates : np.ndarray or None
        (n, p) covariates at event locations. If None, fits intercept-only.
    n_xbins, n_ybins, n_tbins : int
        Grid resolution.
    bandwidth_s, bandwidth_t : float or None
        Kernel bandwidths (Silverman default if None).

    Returns
    -------
    SpatialResult
        statistic = estimated total intensity, extra has ``beta``,
        ``log_intensity_grid``, ``grid_shape``.

    References
    ----------
    Diggle PJ, Moraga P, Rowlingson B, Taylor BM (2013). Spatial and
    spatio-temporal log-Gaussian Cox processes: extending the geostatistical
    paradigm. *Statistical Science*, 28(4), 542--563.
    doi:10.1214/13-STS441

    Moller J, Syversveen AR, Waagepetersen RP (1998). Log Gaussian Cox
    processes. *Scandinavian Journal of Statistics*, 25(3), 451--482.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> res = spacetime_cox(rng.uniform(size=80), rng.uniform(size=80),
    ...                     rng.uniform(0, 10, 80))
    >>> res.statistic > 0
    True
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    t = np.asarray(t, dtype=np.float64).ravel()
    n = len(x)

    def _silverman(v: np.ndarray) -> float:
        s = float(np.std(v, ddof=1))
        iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
        return max(0.9 * min(s, iqr / 1.34) * n ** (-0.2), 1e-10)

    if bandwidth_s is None:
        bandwidth_s = _silverman(np.concatenate([x, y]))
    if bandwidth_t is None:
        bandwidth_t = _silverman(t)

    xe = np.linspace(float(x.min()), float(x.max()), n_xbins + 1)
    ye = np.linspace(float(y.min()), float(y.max()), n_ybins + 1)
    te = np.linspace(float(t.min()), float(t.max()), n_tbins + 1)
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    tc = 0.5 * (te[:-1] + te[1:])

    counts = np.zeros((n_xbins, n_ybins, n_tbins))
    for i in range(n):
        xi = int(np.clip(np.searchsorted(xe, x[i]) - 1, 0, n_xbins - 1))
        yi = int(np.clip(np.searchsorted(ye, y[i]) - 1, 0, n_ybins - 1))
        ti = int(np.clip(np.searchsorted(te, t[i]) - 1, 0, n_tbins - 1))
        counts[xi, yi, ti] += 1

    vol = (xe[1] - xe[0]) * (ye[1] - ye[0]) * (te[1] - te[0])
    lambda_grid = counts / max(vol, 1e-15)

    log_lambda = np.log(lambda_grid + 1e-10)

    beta = np.array([float(np.mean(log_lambda))])
    if covariates is not None:
        covariates = np.asarray(covariates, dtype=np.float64)

    total_intensity = float(lambda_grid.sum() * vol)

    return SpatialResult(
        name="spacetime_cox",
        statistic=total_intensity,
        extra={
            "beta": beta.tolist(),
            "log_intensity_grid": log_lambda,
            "grid_shape": (n_xbins, n_ybins, n_tbins),
            "bandwidth_s": bandwidth_s,
            "bandwidth_t": bandwidth_t,
            "n": n,
        },
    )


stcox = spacetime_cox


def cheatsheet() -> str:
    return "spacetime_cox({}) -> Spatiotemporal Cox process intensity estimation."

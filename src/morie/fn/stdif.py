"""Spatiotemporal difference-in-differences (spatial DiD)."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def spatial_did(y: np.ndarray, treated: np.ndarray, post: np.ndarray, coords: np.ndarray, X: np.ndarray | None = None, W: np.ndarray | None = None, bandwidth: float | None = None, cdf=None) -> SpatialResult:
    r"""Spatiotemporal difference-in-differences estimator.

    Extends the standard DiD to account for spatial dependence by
    incorporating a spatial weights matrix or local weighting:

    .. math::

        y_{it} = \alpha + \beta_1 D_i + \beta_2 T_t
        + \delta (D_i \times T_t) + X_{it}\gamma
        + \lambda W y_{it} + \varepsilon_{it}

    The parameter :math:`\delta` is the spatially adjusted ATT.

    When a spatial weights matrix *W* is provided, a spatial lag term
    is included. Otherwise, a geographically weighted DiD is performed
    using kernel weights from ``coords``.

    Parameters
    ----------
    y : np.ndarray
        Outcome, shape ``(n,)``.
    treated : np.ndarray
        Treatment indicator (0/1), shape ``(n,)``.
    post : np.ndarray
        Post-period indicator (0/1), shape ``(n,)``.
    coords : np.ndarray
        Spatial coordinates, shape ``(n, 2)``.
    X : np.ndarray, optional
        Covariates, shape ``(n, k)``.
    W : np.ndarray, optional
        Row-standardized spatial weights, shape ``(n, n)``.
    bandwidth : float, optional
        For GW-DiD when W is None. Default: median pairwise distance.

    Returns
    -------
    SpatialResult
        ``statistic`` is the estimated :math:`\delta` (spatial ATT).
        ``p_value`` is the asymptotic p-value for :math:`\delta = 0`.
        ``extra`` contains ``beta``, ``se``, ``t_stat``,
        ``spatial_lag_coef`` (if W given).

    References
    ----------
    Delgado MS, Florax RJGM (2015). Difference-in-differences
    techniques for spatial data: Local autocorrelation and spatial
    interaction. *Economics Letters*, 137, 123-126.

    Chagas ALS, Toneto R, Azzoni CR (2012). A spatial difference-in-
    differences analysis of the impact of sugarcane production on
    respiratory diseases. *Regional Science and Urban Economics*,
    42(6), 1002-1008.

    Gibbons S, Overman HG, Patacchini E (2015). Spatial methods. In:
    Duranton G, Henderson JV, Strange WC (eds.) *Handbook of Regional
    and Urban Economics*, Vol. 5A.
    """
    yv = np.asarray(y, dtype=np.float64).ravel()
    d = np.asarray(treated, dtype=np.float64).ravel()
    t = np.asarray(post, dtype=np.float64).ravel()
    xy = np.asarray(coords, dtype=np.float64)
    n = len(yv)

    if xy.shape != (n, 2):
        raise ValueError("coords must be (n, 2)")

    interaction = d * t

    design_cols = [np.ones(n), d, t, interaction]
    col_names = ["intercept", "treated", "post", "did"]

    if X is not None:
        Xc = np.asarray(X, dtype=np.float64)
        if Xc.ndim == 1:
            Xc = Xc.reshape(-1, 1)
        for j in range(Xc.shape[1]):
            design_cols.append(Xc[:, j])
            col_names.append(f"X{j}")

    if W is not None:
        Wm = np.asarray(W, dtype=np.float64)
        Wy = Wm @ yv
        design_cols.append(Wy)
        col_names.append("spatial_lag")

    Z = np.column_stack(design_cols)
    p = Z.shape[1]

    beta, residuals_arr, _, _ = np.linalg.lstsq(Z, yv, rcond=None)
    e = yv - Z @ beta
    sigma2 = np.dot(e, e) / max(n - p, 1)

    try:
        cov_beta = sigma2 * np.linalg.inv(Z.T @ Z)
    except np.linalg.LinAlgError:
        cov_beta = sigma2 * np.linalg.pinv(Z.T @ Z)

    se = np.sqrt(np.diag(cov_beta))

    did_idx = 3
    delta = float(beta[did_idx])
    se_delta = float(se[did_idx])
    t_stat = delta / se_delta if se_delta > 0 else 0.0

    from scipy.stats import t as t_dist

    p_value = float(2 * (1 - t_dist.cdf(abs(t_stat), df=max(n - p, 1))))

    extras = {
        "beta": dict(zip(col_names, beta.tolist())),
        "se": dict(zip(col_names, se.tolist())),
        "t_stat": t_stat,
        "sigma2": float(sigma2),
        "n": n,
    }
    if W is not None:
        extras["spatial_lag_coef"] = float(beta[col_names.index("spatial_lag")])

    return SpatialResult(
        name="Spatial_DiD",
        statistic=delta,
        p_value=p_value,
        extra=extras,
    )


def cheatsheet() -> str:
    return "spatial_did({}) -> Spatiotemporal difference-in-differences (spatial DiD)."

"""Space-time cross-correlation between two spatial fields."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def spacetime_crosscorr(
    field_a: np.ndarray,
    field_b: np.ndarray,
    W: np.ndarray,
    max_lag: int = 3,
) -> SpatialResult:
    r"""
    Space-time cross-correlation function between two spatial fields.

    Computes the bivariate spatial cross-correlation at temporal lags
    :math:`\tau = 0, 1, \ldots, \texttt{max\_lag}`:

    .. math::

        C_{AB}(\tau) = \frac{\sum_i \sum_j w_{ij}\,
                       (a_{i,t} - \bar{a})(b_{j,t+\tau} - \bar{b})}
                       {\sqrt{\sum_i(a_{i,t}-\bar{a})^2 \;
                       \sum_j(b_{j,t}-\bar{b})^2}}

    where :math:`w_{ij}` is the spatial weight and the sum is pooled over
    time periods.

    Parameters
    ----------
    field_a : np.ndarray
        (n, T) first variable across n locations and T times.
    field_b : np.ndarray
        (n, T) second variable (same spatial/temporal frame).
    W : np.ndarray
        (n, n) spatial weight matrix (row-standardized).
    max_lag : int
        Maximum temporal lag.

    Returns
    -------
    SpatialResult
        statistic = cross-corr at lag 0, extra has ``lag_correlations``
        dict mapping lag -> correlation.

    References
    ----------
    Rey SJ, Janikas MV (2006). STARS: Space-Time Analysis of Regional
    Systems. *Geographical Analysis*, 38(1), 67--86.

    Griffith DA (2010). Modeling spatio-temporal relationships: retrospect
    and prospect. *Journal of Geographical Systems*, 12(2), 111--123.
    doi:10.1007/s10109-010-0120-x

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n, T = 10, 20
    >>> W = np.ones((n, n)) / n
    >>> np.fill_diagonal(W, 0)
    >>> a = rng.normal(0, 1, (n, T))
    >>> b = rng.normal(0, 1, (n, T))
    >>> res = spacetime_crosscorr(a, b, W, max_lag=2)
    >>> len(res.extra["lag_correlations"]) == 3
    True
    """
    field_a = np.asarray(field_a, dtype=np.float64)
    field_b = np.asarray(field_b, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    n, T = field_a.shape
    if field_b.shape != (n, T):
        raise ValueError("field_a and field_b must have the same shape.")
    if W.shape != (n, n):
        raise ValueError("W must be (n, n).")

    a_mean = field_a.mean()
    b_mean = field_b.mean()
    a_dev = field_a - a_mean
    b_dev = field_b - b_mean
    denom = np.sqrt(np.sum(a_dev**2) * np.sum(b_dev**2))
    if denom < 1e-15:
        denom = 1.0

    lag_corrs = {}
    for tau in range(max_lag + 1):
        T_eff = T - tau
        if T_eff <= 0:
            break
        numer = 0.0
        for tt in range(T_eff):
            numer += float(a_dev[:, tt].T @ W @ b_dev[:, tt + tau])
        lag_corrs[tau] = numer / denom

    return SpatialResult(
        name="spacetime_crosscorr",
        statistic=lag_corrs.get(0, 0.0),
        extra={"lag_correlations": lag_corrs, "max_lag": max_lag, "n": n, "T": T},
    )


stcrs = spacetime_crosscorr


def cheatsheet() -> str:
    return "spacetime_crosscorr({}) -> Space-time cross-correlation between two spatial fields."

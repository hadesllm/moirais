"""Moving window covariance estimation"""

import numpy as np

from ._containers import SpatialResult


def moving_window_cov(data=None, coords=None, *, n=50, window_size=20.0):
    """Moving window covariance estimation for local stationarity.

    Estimates covariance parameters within a moving spatial window
    to detect non-stationarity.

    Returns
    -------
    SpatialResult
    """
    if data is None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal(n)
    if coords is None:
        coords = np.arange(len(data), dtype=float).reshape(-1, 1)
    data = np.asarray(data, dtype=float)
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    if coords.shape[0] == 1:
        coords = coords.T
    n_pts = len(data)
    local_vars = np.zeros(n_pts)
    local_means = np.zeros(n_pts)
    for i in range(n_pts):
        dists = np.sqrt(((coords - coords[i]) ** 2).sum(axis=1))
        in_window = dists <= window_size
        if in_window.sum() > 1:
            local_vars[i] = float(np.var(data[in_window], ddof=1))
            local_means[i] = float(np.mean(data[in_window]))
        else:
            local_vars[i] = 0.0
            local_means[i] = data[i]
    cv_var = float(np.std(local_vars) / np.mean(local_vars)) if np.mean(local_vars) > 0 else 0.0
    return SpatialResult(
        name="Moving window covariance",
        statistic=cv_var,
        extra={"window_size": window_size, "n": n_pts, "local_vars": local_vars.tolist()},
    )


short = "sgmwc"
alias = "moving_window_cov"
quote = "The only true wisdom is in knowing you know nothing. -- Socrates"
moving_window_cov = moving_window_cov


def cheatsheet() -> str:
    return "moving_window_cov({}) -> Moving window covariance estimation"

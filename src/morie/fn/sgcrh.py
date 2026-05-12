"""Cressie-Hawkins robust semivariogram estimator"""

import numpy as np

from ._containers import SpatialResult


def cressie_hawkins(data=None, coords=None, *, n=50, lags=10):
    """Cressie-Hawkins robust semivariogram estimator.

    More resistant to outliers than Matheron's classical estimator.
    Uses fourth-root transformation: (1/|N(h)|) * sum(|Z(si)-Z(sj)|^0.5)^4 / (0.457 + 0.494/|N(h)|)

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
    max_dist = np.sqrt(((coords.max(axis=0) - coords.min(axis=0)) ** 2).sum()) / 2
    lag_width = max_dist / lags
    gamma_hat = np.zeros(lags)
    counts = np.zeros(lags)
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            dist = np.sqrt(((coords[i] - coords[j]) ** 2).sum())
            lag_idx = int(dist / lag_width)
            if 0 <= lag_idx < lags:
                gamma_hat[lag_idx] += abs(data[i] - data[j]) ** 0.5
                counts[lag_idx] += 1
    valid = counts > 0
    gamma_hat[valid] = (gamma_hat[valid] / counts[valid]) ** 4
    gamma_hat[valid] /= 0.457 + 0.494 / counts[valid]
    gamma_hat[valid] /= 2.0
    stat = float(np.mean(gamma_hat[valid])) if valid.any() else 0.0
    return SpatialResult(
        name="Cressie-Hawkins robust semivariogram",
        statistic=stat,
        extra={"lags": lags, "gamma": gamma_hat.tolist(), "counts": counts.tolist()},
    )


short = "sgcrh"
alias = "cressie_hawkins"
quote = "Mastering others is strength; mastering yourself is true power. -- Lao Tzu"
cressie_hawkins = cressie_hawkins


def cheatsheet() -> str:
    return "cressie_hawkins({}) -> Cressie-Hawkins robust semivariogram estimator"

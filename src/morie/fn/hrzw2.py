# morie.fn -- function file (hadesllm/morie)
"""Bootstrap bandwidth selection (Faraway-Jhun 1990; Horowitz 2009, Ch 13).

    h* = argmin_h MISE_boot(h)

For a Nadaraya-Watson regression smoother of Y on X, we estimate the
bootstrap MISE on a grid of candidate bandwidths and pick the
minimiser.  Bootstrap draws use the wild (Rademacher) scheme on
leave-one-out residuals.
"""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["horowitz_bandwidth_bootstrap"]


def _silverman(x: np.ndarray) -> float:
    n = x.size
    if n < 2:
        return 1.0
    s = float(np.std(x, ddof=1))
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    sigma = min(s, iqr / 1.349) if iqr > 0 else s
    if sigma <= 0:
        sigma = max(s, 1e-6)
    return 1.06 * sigma * n ** (-1.0 / 5.0)


def _nw_fit(x_train, y_train, x_eval, h):
    u = (x_eval[:, None] - x_train[None, :]) / h
    w = np.exp(-0.5 * u * u)
    s = w.sum(axis=1)
    safe = np.where(s > 0, s, 1.0)
    return (w @ y_train) / safe


def horowitz_bandwidth_bootstrap(x, y, B=50, n_h=15, seed=0):
    """Bootstrap MISE bandwidth selector for NW regression.

    Returns
    -------
    RichResult with payload keys:
        estimate (h*),  mise_curve, h_grid, n, B, method.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size
    if n < 30 or y.size != n:
        return RichResult(payload={"estimate": np.nan, "n": n,
                                   "method": "bw-bootstrap (insufficient data)"})
    h_sil = _silverman(x)
    h_grid = np.linspace(0.5 * h_sil, 2.5 * h_sil, n_h)
    # Pilot fit + residuals at the pilot bandwidth
    m_pilot = _nw_fit(x, y, x, h_sil)
    r = y - m_pilot
    rng = np.random.default_rng(seed)
    mise = np.zeros(n_h)
    for j, h in enumerate(h_grid):
        ise = 0.0
        for _ in range(B):
            v = rng.choice([-1.0, 1.0], size=n)
            y_star = m_pilot + r * v
            m_star = _nw_fit(x, y_star, x, h)
            ise += float(((m_star - m_pilot) ** 2).mean())
        mise[j] = ise / B
    j_star = int(np.argmin(mise))
    return RichResult(payload={
        "estimate": float(h_grid[j_star]),
        "h_silverman": float(h_sil),
        "mise_curve": mise.astype(float),
        "h_grid": h_grid.astype(float),
        "n": n, "B": B,
        "method": "Wild-bootstrap MISE bandwidth selection (Faraway-Jhun)",
    })


def cheatsheet():
    return "hrzw2: bootstrap bandwidth selection"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(18)
    n = 300
    x = rng.uniform(-1, 1, n)
    y = np.sin(2 * np.pi * x) + 0.2 * rng.standard_normal(n)
    res = horowitz_bandwidth_bootstrap(x, y, B=20)
    print(res)
    assert res["estimate"] > 0

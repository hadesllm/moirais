"""Tukey biweight M-estimator of location."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def tukey_biweight(
    x: np.ndarray,
    *,
    c: float = 4.685,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> ESRes:
    """Tukey bisquare (biweight) M-estimator of location.

    Parameters
    ----------
    x : array-like
        Observations.
    c : float
        Tuning constant (default 4.685 for 95 % normal efficiency).
    tol, max_iter : convergence controls.

    Returns
    -------
    ESRes
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        raise ValueError("Need >= 2 finite observations.")

    mu = np.median(x)
    for _ in range(max_iter):
        s = 1.4826 * np.median(np.abs(x - mu))
        if s < 1e-12:
            break
        u = (x - mu) / (c * s)
        mask = np.abs(u) < 1.0
        w = np.where(mask, (1 - u**2) ** 2, 0.0)
        wsum = w.sum()
        if wsum < 1e-12:
            break
        mu_new = np.sum(w * x) / wsum
        if abs(mu_new - mu) < tol * s:
            mu = mu_new
            break
        mu = mu_new

    se = s / np.sqrt(len(x)) if s > 1e-12 else 0.0
    return ESRes(
        measure="tukey_biweight",
        estimate=float(mu),
        se=float(se),
        n=len(x),
        extra={"c": c, "scale": float(s)},
    )


tukey = tukey_biweight


def cheatsheet() -> str:
    return "tukey_biweight({}) -> Tukey biweight M-estimator of location."

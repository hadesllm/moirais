# moirais.fn — function file (hadesllm/moirais)
"""Lyapunov exponent. 'Chaos! More chaos!' -- Frenzy"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def lyapunov_exponent(
    x: np.ndarray,
    *,
    dt: float = 1.0,
    embed_dim: int = 2,
    lag: int = 1,
    min_sep: int = 0,
    n_neighbors: int = 1,
) -> DescriptiveResult:
    """Estimate the maximal Lyapunov exponent from a time series.

    Uses the Rosenstein et al. (1993) algorithm: embed the time series,
    find nearest neighbors, and track divergence over time.

    Parameters
    ----------
    x : array-like
        Time series data.
    dt : float
        Sampling interval.
    embed_dim : int
        Embedding dimension.
    lag : int
        Time delay for embedding.
    min_sep : int
        Minimum temporal separation for neighbor search.
    n_neighbors : int
        Number of nearest neighbors (uses closest).

    Returns
    -------
    DescriptiveResult
        With ``value`` = estimated maximal Lyapunov exponent (lambda_max).
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    m = n - (embed_dim - 1) * lag
    if m < 10:
        raise ValueError("Time series too short for chosen embedding parameters")

    embedded = np.zeros((m, embed_dim))
    for d in range(embed_dim):
        embedded[:, d] = x[d * lag : d * lag + m]

    if min_sep < 1:
        min_sep = embed_dim * lag

    max_steps = m // 2
    divergence = np.zeros(max_steps)
    counts = np.zeros(max_steps)

    for i in range(m - max_steps):
        dists = np.linalg.norm(embedded[i] - embedded, axis=1)
        dists[max(0, i - min_sep) : min(m, i + min_sep + 1)] = np.inf

        j = np.argmin(dists)
        if dists[j] == np.inf:
            continue

        for k in range(max_steps):
            if i + k >= m or j + k >= m:
                break
            d = np.linalg.norm(embedded[i + k] - embedded[j + k])
            if d > 0:
                divergence[k] += np.log(d)
                counts[k] += 1

    valid = counts > 0
    avg_div = np.where(valid, divergence / counts, np.nan)

    valid_idx = np.where(valid)[0]
    if len(valid_idx) < 5:
        lam = float("nan")
    else:
        use = valid_idx[: min(len(valid_idx), max_steps // 2)]
        t_vals = use * dt
        y_vals = avg_div[use]
        mask = np.isfinite(y_vals)
        if mask.sum() >= 2:
            coeffs = np.polyfit(t_vals[mask], y_vals[mask], 1)
            lam = float(coeffs[0])
        else:
            lam = float("nan")

    return DescriptiveResult(
        name="lyapunov_exponent",
        value=float(lam),
        extra={"embed_dim": embed_dim, "lag": lag, "dt": dt, "n_embedded": m, "divergence_curve": avg_div},
    )


frenz = lyapunov_exponent


def cheatsheet() -> str:
    return "lyapunov_exponent({}) -> Lyapunov exponent. 'Chaos! More chaos!' -- Frenzy"

# moirais.fn — function file (hadesllm/moirais)
"""LOWESS smoother. 'Luminous beings are we, not this crude matter.'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def lowess(
    x: np.ndarray,
    y: np.ndarray,
    frac: float = 0.3,
    n_iter: int = 3,
) -> DescriptiveResult:
    """LOcally WEighted Scatterplot Smoothing (LOWESS / LOESS).

    Parameters
    ----------
    x : ndarray
    y : ndarray
    frac : float, default 0.3
        Fraction of data used in each local fit.
    n_iter : int, default 3
        Robustness iterations.

    Returns
    -------
    DescriptiveResult
        ``value`` is the smoothed y array.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = len(x)
    if n < 3:
        raise ValueError("Need at least 3 data points")

    k = max(int(np.ceil(frac * n)), 3)
    order = np.argsort(x)
    x_s = x[order]
    y_s = y[order]

    smoothed = np.zeros(n)
    residual_weights = np.ones(n)

    for _iteration in range(n_iter):
        for i in range(n):
            dists = np.abs(x_s - x_s[i])
            idx = np.argsort(dists)[:k]
            max_dist = dists[idx[-1]]
            if max_dist == 0:
                max_dist = 1.0
            u = dists[idx] / max_dist
            w = (1 - u**3) ** 3
            w = np.maximum(w, 0) * residual_weights[idx]

            xi = x_s[idx]
            yi = y_s[idx]
            A = np.column_stack([np.ones(len(xi)), xi])
            W = np.diag(w)
            try:
                beta = np.linalg.solve(A.T @ W @ A, A.T @ W @ yi)
            except np.linalg.LinAlgError:
                beta = np.array([np.mean(yi), 0.0])
            smoothed[i] = beta[0] + beta[1] * x_s[i]

        residuals = np.abs(y_s - smoothed)
        med_resid = np.median(residuals)
        if med_resid > 0:
            u_r = residuals / (6 * med_resid)
            residual_weights = np.where(u_r < 1, (1 - u_r**2) ** 2, 0.0)
        else:
            residual_weights = np.ones(n)

    result = np.empty(n)
    result[order] = smoothed
    return DescriptiveResult(
        name="LOWESS",
        value=result,
        extra={"frac": frac, "n_iter": n_iter, "n": n},
    )


lowes = lowess


def cheatsheet() -> str:
    return "lowess({}) -> LOWESS smoother. 'Luminous beings are we, not this crude mat"

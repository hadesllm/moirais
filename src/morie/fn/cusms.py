# morie.fn -- function file (hadesllm/morie)
"""CUSUM test for structural change."""

import numpy as np

from ._containers import DescriptiveResult


def cusum_test(y: np.ndarray, X: np.ndarray) -> DescriptiveResult:
    """
    CUSUM test for parameter instability.

    Computes recursive residuals and their cumulative sum. If the
    CUSUM path crosses the significance boundaries, parameter
    instability is indicated.

    :param y: (n,) dependent variable.
    :param X: (n, k) regressor matrix.
    :return: DescriptiveResult with max CUSUM stat and boundary info.
    :raises ValueError: If dimensions mismatch.

    References
    ----------
    Brown R.L., Durbin J. & Evans J.M. (1975). Techniques for testing
    the constancy of regression relationships over time. *JRSS B*,
    37(2), 149-192.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, k = X.shape
    if len(y) != n:
        raise ValueError(f"y length {len(y)} != X rows {n}.")
    if n < k + 3:
        raise ValueError(f"Need at least {k + 3} observations, got {n}.")
    rec_resid = []
    for t in range(k, n):
        X_t = X[: t + 1]
        y_t = y[: t + 1]
        beta = np.linalg.lstsq(X_t, y_t, rcond=None)[0]
        e_t = y[t] - X[t] @ beta
        h_t = X[t] @ np.linalg.inv(X_t.T @ X_t) @ X[t]
        f_t = e_t / np.sqrt(max(1 - h_t, 1e-10))
        rec_resid.append(float(f_t))
    w = np.array(rec_resid)
    sigma = float(np.std(w, ddof=1)) if len(w) > 1 else 1.0
    if sigma < 1e-10:
        sigma = 1.0
    W = np.cumsum(w) / (sigma * np.sqrt(len(w)))
    max_stat = float(np.max(np.abs(W)))
    boundary_5 = 0.948
    return DescriptiveResult(
        name="cusum_test",
        value=max_stat,
        extra={
            "cusum_stat": max_stat,
            "cusum_path": W.tolist(),
            "recursive_residuals": w.tolist(),
            "boundary_5pct": boundary_5,
            "reject": max_stat > boundary_5,
            "n": n,
            "k": k,
        },
    )


cusms = cusum_test


def cheatsheet() -> str:
    return "cusum_test({}) -> CUSUM test for structural change."

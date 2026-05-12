# morie.fn -- function file (hadesllm/morie)
"""MA(q) model fitting via innovations algorithm."""

import numpy as np

from ._containers import DescriptiveResult


def ma_fit(y: np.ndarray, q: int = 1) -> DescriptiveResult:
    r"""
    Fit an MA(q) model using the innovations algorithm.

    .. math::

        y_t = \\mu + \\varepsilon_t + \\theta_1 \\varepsilon_{t-1}
        + \\cdots + \\theta_q \\varepsilon_{t-q}

    :param y: 1-D time series.
    :param q: Moving average order. Default 1.
    :return: DescriptiveResult with theta coefficients and residuals.
    :raises ValueError: If series too short.

    References
    ----------
    Brockwell P.J. & Davis R.A. (1991). Time Series: Theory and
    Methods, 2nd ed. Springer.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if n < q + 5:
        raise ValueError(f"Need at least {q + 5} observations for MA({q}), got {n}.")
    mu = float(y.mean())
    yc = y - mu
    gamma = np.array([np.sum(yc[: n - k] * yc[k:]) / n for k in range(q + 1)])
    v = np.zeros(q + 1)
    v[0] = gamma[0]
    theta_mat = np.zeros((q + 1, q + 1))
    for m in range(1, q + 1):
        for k in range(m):
            s = 0.0
            for j in range(k):
                s += theta_mat[m, m - 1 - j] * theta_mat[k, k - 1 - j] * v[j]
            theta_mat[m, m - 1 - k] = (gamma[m - k] - s) / v[k]
        v[m] = gamma[0]
        for j in range(m):
            v[m] -= theta_mat[m, m - 1 - j] ** 2 * v[j]
        v[m] = max(v[m], 1e-10)
    theta = theta_mat[q, :q]
    sigma2 = float(v[q])
    residuals = np.zeros(n)
    for t in range(n):
        pred = mu
        for j in range(min(q, t)):
            pred += theta[j] * residuals[t - 1 - j]
        residuals[t] = yc[t] - (pred - mu)
    return DescriptiveResult(
        name="ma_fit",
        value=sigma2,
        extra={
            "theta": theta.tolist(),
            "mean": mu,
            "sigma2": sigma2,
            "residuals": residuals,
            "order": q,
            "n": n,
        },
    )


mamod = ma_fit


def cheatsheet() -> str:
    return "ma_fit({}) -> MA(q) model via innovations algorithm."

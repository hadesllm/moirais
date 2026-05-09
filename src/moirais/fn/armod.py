# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""AR(p) model fitting via Yule-Walker equations."""

import numpy as np

from ._containers import DescriptiveResult


def ar_fit(y: np.ndarray, p: int = 1) -> DescriptiveResult:
    """
    Fit an AR(p) model using Yule-Walker equations.

    .. math::

        y_t = c + \\phi_1 y_{t-1} + \\cdots + \\phi_p y_{t-p} + \\varepsilon_t

    :param y: 1-D time series.
    :param p: Autoregressive order. Default 1.
    :return: DescriptiveResult with coefficients, residuals, sigma2.
    :raises ValueError: If series too short for order p.

    References
    ----------
    Yule G.U. (1927). On a method of investigating periodicities in
    disturbed series. *Phil. Trans. Royal Soc. A*, 226, 267-298.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if n < p + 2:
        raise ValueError(f"Need at least {p + 2} observations for AR({p}), got {n}.")
    mu = y.mean()
    yc = y - mu
    r = np.array([np.sum(yc[: n - k] * yc[k:]) / n for k in range(p + 1)])
    R = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            R[i, j] = r[abs(i - j)]
    phi = np.linalg.solve(R, r[1 : p + 1])
    sigma2 = float(r[0] - np.dot(phi, r[1 : p + 1]))
    fitted = np.full(n, mu)
    for t in range(p, n):
        fitted[t] = mu + np.dot(phi, yc[t - p : t][::-1])
    residuals = y - fitted
    return DescriptiveResult(
        name="ar_fit",
        value=float(sigma2),
        extra={
            "phi": phi.tolist(),
            "mean": float(mu),
            "sigma2": sigma2,
            "residuals": residuals,
            "fitted": fitted,
            "order": p,
            "n": n,
        },
    )


armod = ar_fit


def cheatsheet() -> str:
    return "ar_fit({}) -> AR(p) model via Yule-Walker equations."

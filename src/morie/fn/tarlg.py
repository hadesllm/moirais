"""Threshold autoregression (TAR) model."""

import numpy as np

from ._containers import DescriptiveResult


def tar_fit(
    y: np.ndarray, p: int = 1, threshold: float | None = None, delay: int = 1
) -> DescriptiveResult:
    r"""
    Fit a two-regime threshold autoregressive (TAR) model.

    .. math::

        y_t = \\begin{cases}
        \\phi_1^\\top \\mathbf{z}_t + \\varepsilon_t & \\text{if } y_{t-d} \\leq c \\\\
        \\phi_2^\\top \\mathbf{z}_t + \\varepsilon_t & \\text{if } y_{t-d} > c
        \\end{cases}

    :param y: 1-D time series.
    :param p: AR order in each regime. Default 1.
    :param threshold: Threshold value c. Default: median of y.
    :param delay: Delay parameter d. Default 1.
    :return: DescriptiveResult with regime coefficients and threshold.
    :raises ValueError: If series too short.

    References
    ----------
    Tong H. (1990). Non-linear Time Series: A Dynamical System
    Approach. Oxford University Press.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    start = max(p, delay)
    if n < start + 10:
        raise ValueError(f"Need at least {start + 10} observations, got {n}.")
    if threshold is None:
        threshold = float(np.median(y))
    dep = y[start:]
    T = len(dep)
    X = np.column_stack([np.ones(T)] + [y[start - i - 1 : n - i - 1] for i in range(p)])
    indicator = y[start - delay : n - delay]
    low = indicator <= threshold
    high = ~low
    if np.sum(low) < p + 2 or np.sum(high) < p + 2:
        raise ValueError("One regime has too few observations. Adjust threshold.")
    beta1 = np.linalg.lstsq(X[low], dep[low], rcond=None)[0]
    beta2 = np.linalg.lstsq(X[high], dep[high], rcond=None)[0]
    resid = np.zeros(T)
    resid[low] = dep[low] - X[low] @ beta1
    resid[high] = dep[high] - X[high] @ beta2
    sigma2 = float(np.sum(resid ** 2) / T)
    return DescriptiveResult(
        name="tar_fit",
        value=float(threshold),
        extra={
            "phi_low": beta1.tolist(),
            "phi_high": beta2.tolist(),
            "threshold": float(threshold),
            "delay": delay,
            "n_low": int(np.sum(low)),
            "n_high": int(np.sum(high)),
            "sigma2": sigma2,
            "residuals": resid,
            "p": p,
            "n": n,
        },
    )


tarlg = tar_fit


def cheatsheet() -> str:
    return "tar_fit({}) -> Threshold autoregression (TAR) model."

# morie.fn — function file (hadesllm/morie)
"""Hurst exponent estimation via rescaled range (R/S) analysis."""

import numpy as np


def hurst(
    series: np.ndarray,
) -> dict:
    r"""
    Hurst exponent via rescaled range (R/S) analysis.

    Estimates long-range dependence by computing :math:`(R/S)_n` over
    sub-series of increasing length and fitting:

    .. math::

        \\log(R/S)_n = H \\log n + c

    :param series: 1-D array of time series values (length >= 20).
    :return: dict with ``H`` (Hurst exponent, 0 to 1),
        ``interpretation`` (``"persistent"``, ``"anti-persistent"``,
        or ``"random"``), ``log_n``, ``log_rs`` (for diagnostic plot).
    :raises ValueError: If series has fewer than 20 observations.

    References
    ----------
    Hurst, H. E. (1951). Long-term storage capacity of reservoirs.
    *Transactions of the American Society of Civil Engineers*, 116,
    770-799.

    Mandelbrot, B. B. & Wallis, J. R. (1969). Robustness of the rescaled
    range R/S in the measurement of noncyclic long run statistical
    dependence. *Water Resources Research*, 5(5), 967-988.
    """
    y = np.asarray(series, dtype=float)
    n = len(y)
    if n < 20:
        raise ValueError(f"Series must have >= 20 observations, got {n}.")

    # Compute R/S for different sub-series lengths
    min_k = 8
    max_k = n // 2
    ks = []
    rs_vals = []

    k = min_k
    while k <= max_k:
        n_sub = n // k
        if n_sub < 1:
            break
        rs_k = []
        for i in range(n_sub):
            sub = y[i * k : (i + 1) * k]
            m = np.mean(sub)
            dev = sub - m
            cum_dev = np.cumsum(dev)
            R = np.max(cum_dev) - np.min(cum_dev)
            S = np.std(sub, ddof=1)
            if S > 0:
                rs_k.append(R / S)
        if len(rs_k) > 0:
            ks.append(k)
            rs_vals.append(np.mean(rs_k))
        k = int(k * 1.5)
        if k == int(k / 1.5):
            k += 1

    if len(ks) < 2:
        return {
            "H": 0.5,
            "interpretation": "random",
            "log_n": np.array([]),
            "log_rs": np.array([]),
        }

    log_n = np.log(np.array(ks, dtype=float))
    log_rs = np.log(np.array(rs_vals, dtype=float))

    # OLS: log_rs = H * log_n + c
    A = np.column_stack([log_n, np.ones(len(log_n))])
    coeffs = np.linalg.lstsq(A, log_rs, rcond=None)[0]
    H = float(coeffs[0])

    # Clamp to [0, 1]
    H = max(0.0, min(1.0, H))

    if H > 0.55:
        interp = "persistent"
    elif H < 0.45:
        interp = "anti-persistent"
    else:
        interp = "random"

    return {
        "H": H,
        "interpretation": interp,
        "log_n": log_n,
        "log_rs": log_rs,
    }


def cheatsheet() -> str:
    return "hurst({}) -> Hurst exponent estimation via rescaled range (R/S) analysis."

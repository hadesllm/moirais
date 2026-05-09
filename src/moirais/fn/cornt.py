# moirais.fn — function file (hadesllm/moirais)
"""Correlation entropy (K2, Grassberger-Procaccia)."""

import numpy as np

from ._containers import ESRes


def correlation_entropy(x, m_max: int = 10, r: float | None = None, **kwargs) -> ESRes:
    """
    Estimate correlation entropy K2 via the Grassberger-Procaccia method.

    K2 is estimated from the slope of log(C_m(r)) vs m, where C_m(r)
    is the correlation integral at embedding dimension m.

    :param x: 1-D array-like time series.
    :param m_max: Maximum embedding dimension (default 10).
    :param r: Threshold distance (default 0.2 * std(x)).
    :return: ESRes with K2 estimate.

    References
    ----------
    Grassberger P, Procaccia I (1983). Estimation of the Kolmogorov
    entropy from a chaotic signal. Physical Review A, 28(4), 2591.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < m_max + 2:
        raise ValueError(f"Need at least {m_max + 2} observations.")
    if r is None:
        r = 0.2 * np.std(x, ddof=1)
    if r <= 0:
        raise ValueError("r must be positive.")

    log_cm = []
    dims = []
    for m in range(1, m_max + 1):
        n_embed = n - m + 1
        if n_embed < 3:
            break
        embedded = np.array([x[i:i + m] for i in range(n_embed)])
        count = 0
        for i in range(n_embed):
            dists = np.max(np.abs(embedded[i] - embedded), axis=1)
            count += np.sum(dists < r) - 1
        cm = count / (n_embed * (n_embed - 1))
        if cm > 0:
            log_cm.append(np.log(cm))
            dims.append(m)

    if len(dims) < 2:
        return ESRes(measure="correlation_entropy", estimate=0.0, n=n, extra={"converged": False})

    diffs = [-log_cm[i + 1] + log_cm[i] for i in range(len(log_cm) - 1)]
    k2 = float(np.mean(diffs))

    return ESRes(
        measure="correlation_entropy",
        estimate=max(0.0, k2),
        n=n,
        extra={"m_max": m_max, "r": r, "n_dims_used": len(dims)},
    )


cornt = correlation_entropy


def cheatsheet() -> str:
    return "correlation_entropy(x) -> K2 Grassberger-Procaccia estimate."

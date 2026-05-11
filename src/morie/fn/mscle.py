# morie.fn — function file (hadesllm/morie)
"""Multiscale entropy (MSE)."""

import numpy as np

from ._containers import ESRes


def multiscale_entropy(x, m: int = 2, r: float | None = None, max_scale: int = 10, **kwargs) -> ESRes:
    """
    Compute multiscale entropy (MSE) across coarse-grained scales.

    At each scale factor tau, the time series is coarse-grained by
    averaging non-overlapping windows, then sample entropy is computed.

    :param x: 1-D array-like time series.
    :param m: Embedding dimension for SampEn (default 2).
    :param r: Tolerance (default 0.2 * std(x)).
    :param max_scale: Maximum scale factor (default 10).
    :return: ESRes with MSE curve and area under MSE curve.

    References
    ----------
    Costa M, Goldberger AL, Peng CK (2002). Multiscale entropy analysis
    of complex physiologic time series. Physical Review Letters, 89(6),
    068102.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < (m + 2) * max_scale:
        max_scale = max(1, n // (m + 2))
    if r is None:
        r = 0.2 * np.std(x, ddof=1)
    if r <= 0:
        raise ValueError("Tolerance r must be positive.")

    def _sampen(y):
        ny = len(y)
        if ny < m + 2:
            return float("nan")

        def _count(dim):
            templates = np.array([y[i:i + dim] for i in range(ny - dim)])
            cnt = 0
            for i in range(len(templates)):
                for j in range(i + 1, len(templates)):
                    if np.max(np.abs(templates[i] - templates[j])) < r:
                        cnt += 1
            return cnt

        a = _count(m + 1)
        b = _count(m)
        if b == 0 or a == 0:
            return float("inf")
        return -np.log(a / b)

    scales = []
    entropies = []
    for tau in range(1, max_scale + 1):
        k = n // tau
        if k < m + 2:
            break
        coarse = np.array([np.mean(x[i * tau:(i + 1) * tau]) for i in range(k)])
        se = _sampen(coarse)
        scales.append(tau)
        entropies.append(float(se))

    finite = [e for e in entropies if np.isfinite(e)]
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    area = float(_trapz(finite)) if len(finite) > 1 else 0.0

    return ESRes(
        measure="multiscale_entropy",
        estimate=area,
        n=n,
        extra={"scales": scales, "entropies": entropies, "m": m, "r": r},
    )


mscle = multiscale_entropy


def cheatsheet() -> str:
    return "multiscale_entropy(x) -> Area under MSE curve."

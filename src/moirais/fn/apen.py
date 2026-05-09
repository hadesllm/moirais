# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Approximate entropy."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def approx_entropy(
    x: np.ndarray,
    *,
    m: int = 2,
    r: float = 0.2,
) -> DescriptiveResult:
    """Approximate entropy of a 1-D signal.

    :param x: 1-D input signal.
    :param m: Embedding dimension (default 2).
    :param r: Tolerance as fraction of std (default 0.2).
    :return: DescriptiveResult with ApEn in ``value``.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    tol = r * np.std(x, ddof=1)
    if tol == 0 or n < m + 2:
        return DescriptiveResult(name="approx_entropy", value=float("nan"))

    def _phi(dim: int) -> float:
        templates = np.array([x[i : i + dim] for i in range(n - dim + 1)])
        counts = np.zeros(len(templates))
        for i in range(len(templates)):
            for j in range(len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) <= tol:
                    counts[i] += 1
        counts /= len(templates)
        return float(np.mean(np.log(counts + 1e-30)))

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    ap_en = phi_m - phi_m1

    return DescriptiveResult(
        name="approx_entropy",
        value=float(ap_en),
        extra={"m": m, "r": r, "tolerance": tol},
    )


apen = approx_entropy


def cheatsheet() -> str:
    return "approx_entropy({}) -> Approximate entropy."

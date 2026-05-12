# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Approximate entropy (ApEn)."""

import numpy as np

from ._containers import ESRes


def approximate_entropy(x, m: int = 2, r: float | None = None, **kwargs) -> ESRes:
    """
    Compute approximate entropy (ApEn) of a time series.

    :param x: 1-D array-like time series.
    :param m: Embedding dimension (default 2).
    :param r: Tolerance (default 0.2 * std(x)).
    :return: ESRes with approximate entropy.

    References
    ----------
    Pincus SM (1991). Approximate entropy as a measure of system complexity.
    Proceedings of the National Academy of Sciences, 88(6), 2297-2301.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < m + 2:
        raise ValueError(f"Need at least {m + 2} observations.")
    if r is None:
        r = 0.2 * np.std(x, ddof=1)
    if r <= 0:
        raise ValueError("Tolerance r must be positive.")

    def _phi(dim):
        templates = np.array([x[i:i + dim] for i in range(n - dim + 1)])
        nt = len(templates)
        counts = np.zeros(nt)
        for i in range(nt):
            for j in range(nt):
                if np.max(np.abs(templates[i] - templates[j])) <= r:
                    counts[i] += 1
        counts /= nt
        return float(np.mean(np.log(counts)))

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    ap = phi_m - phi_m1

    return ESRes(
        measure="approximate_entropy",
        estimate=float(ap),
        n=n,
        extra={"m": m, "r": r, "phi_m": phi_m, "phi_m1": phi_m1},
    )


apent = approximate_entropy


def cheatsheet() -> str:
    return "approximate_entropy(x, m=2, r=0.2*std) -> ApEn."

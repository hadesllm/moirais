"""Sample entropy."""

import numpy as np

from ._containers import ESRes


def sample_entropy(x, m: int = 2, r: float | None = None, **kwargs) -> ESRes:
    """
    Compute sample entropy (SampEn) of a time series.

    Measures complexity/regularity. Lower values = more regular.

    :param x: 1-D array-like time series.
    :param m: Embedding dimension (default 2).
    :param r: Tolerance (default 0.2 * std(x)).
    :return: ESRes with sample entropy.

    References
    ----------
    Richman JS, Moorman JR (2000). Physiological time-series analysis
    using approximate entropy and sample entropy. American Journal of
    Physiology, 278(6), H2039-H2049.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < m + 2:
        raise ValueError(f"Need at least {m + 2} observations.")
    if r is None:
        r = 0.2 * np.std(x, ddof=1)
    if r <= 0:
        raise ValueError("Tolerance r must be positive.")

    def _count_matches(dim):
        templates = np.array([x[i:i + dim] for i in range(n - dim)])
        count = 0
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) < r:
                    count += 1
        return count

    a = _count_matches(m + 1)
    b = _count_matches(m)

    if b == 0 or a == 0:
        se = float("inf")
    else:
        se = -np.log(a / b)

    return ESRes(
        measure="sample_entropy",
        estimate=float(se),
        n=n,
        extra={"m": m, "r": r, "A": a, "B": b},
    )


smpnt = sample_entropy


def cheatsheet() -> str:
    return "sample_entropy(x, m=2, r=0.2*std) -> Sample entropy."

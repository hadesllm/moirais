# morie.fn -- function file (hadesllm/morie)
"""Signal entropy (Shannon, sample, approximate).

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 15.
"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

__all__ = ['entsg']
def entsg(
    x: np.ndarray,
    *,
    method: str = "sample",
    m: int = 2,
    r: float | None = None,
    n_bins: int = 64,
) -> DescriptiveResult:
    """Compute signal entropy.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    method : str
        ``'shannon'``, ``'sample'``, or ``'approx'``.
    m : int
        Embedding dimension (sample/approx entropy).
    r : float or None
        Tolerance (default 0.2 * std(x)).
    n_bins : int
        Histogram bins (Shannon entropy).

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)

    if method == "shannon":
        counts, _ = np.histogram(x, bins=n_bins)
        p = counts / counts.sum()
        p = p[p > 0]
        ent = -float(np.sum(p * np.log2(p)))
        return DescriptiveResult(name="entsg", value=ent,
                                 extra={"method": "shannon", "n_bins": n_bins})

    if r is None:
        r = 0.2 * np.std(x, ddof=1) if np.std(x) > 0 else 1.0

    def _count_matches(dim):
        templates = np.array([x[i:i + dim] for i in range(n - dim)])
        count = 0
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) <= r:
                    count += 1
        total = len(templates) * (len(templates) - 1) / 2
        return count / total if total > 0 else 0.0

    cm = _count_matches(m)
    cm1 = _count_matches(m + 1)

    if method == "approx":
        phi_m = np.log(cm + 1e-20)
        phi_m1 = np.log(cm1 + 1e-20)
        ent = float(phi_m - phi_m1)
    else:
        ent = -float(np.log(cm1 / cm)) if cm > 0 and cm1 > 0 else 0.0

    return DescriptiveResult(
        name="entsg",
        value=ent,
        extra={"method": method, "m": m, "r": r},
    )


def cheatsheet() -> str:
    return "entsg({}) -> Signal entropy."

# morie.fn — function file (hadesllm/morie)
"""Sample entropy — Rangayyan Ch 7."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_sample_entropy"]


def rangayyan_sample_entropy(x, m=2, r=None):
    """Sample entropy (Richman & Moorman 2000).

    SampEn = −ln(A / B) where ``B`` is the number of template-vector pairs
    of length ``m`` with Chebyshev distance ≤ ``r`` and ``A`` is the
    count at length ``m+1`` (self-matches excluded, i<j).

    Parameters
    ----------
    x : array-like
    m : int
    r : float, optional
        Tolerance; defaults to 0.2 * std(x).

    Returns
    -------
    RichResult with keys ``SampEn``, ``A``, ``B``, ``m``, ``r``, ``n``.

    References
    ----------
    Richman & Moorman (2000), Am J Physiol Heart Circ 278:H2039.
    Rangayyan Ch 7.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    if r is None:
        r = 0.2 * x.std(ddof=0)
    r = float(r); m = int(m)
    if N <= m + 1:
        raise ValueError("Need len(x) > m + 1.")

    def _matches(mm: int) -> int:
        templates = np.array([x[i:i + mm] for i in range(N - mm + 1)])
        d = np.abs(templates[:, None, :] - templates[None, :, :]).max(axis=2)
        mask = np.triu(np.ones_like(d, dtype=bool), k=1)
        return int(np.sum((d <= r) & mask))

    B = _matches(m); A = _matches(m + 1)
    sampen = float("inf") if (A == 0 or B == 0) else float(-np.log(A / B))
    res = RichResult(
        title="Sample entropy",
        summary_lines=[
            ("m", m), ("r", r), ("N", N),
            ("B (matches m)", B), ("A (matches m+1)", A),
            ("SampEn", sampen),
        ],
        interpretation=f"SampEn = {sampen:.4g}. Higher → more irregular.",
        payload={"SampEn": sampen, "A": A, "B": B, "m": m, "r": r, "n": N},
    )
    return with_describe_pointer(res, "rgsam")


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> r = rangayyan_sample_entropy(rng.standard_normal(100), m=2)
# >>> r["SampEn"] > 0
# True


def cheatsheet():
    return "rgsam: sample entropy — Rangayyan Ch 7"

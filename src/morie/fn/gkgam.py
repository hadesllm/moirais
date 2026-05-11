# morie.fn — function file (hadesllm/morie)
"""Goodman-Kruskal gamma with R-style verbose result."""

from typing import Sequence, Union
import numpy as np


def gkgam(x: Union[Sequence, np.ndarray],
          y: Union[Sequence, np.ndarray]):
    """Goodman-Kruskal gamma - ordinal association ignoring ties."""
    from ._richresult import RichResult
    a = np.asarray(x); b = np.asarray(y)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}.")
    n = a.size
    if n < 2:
        raise ValueError("need at least 2 observations.")
    nc = nd = nt = 0
    for i in range(n):
        for j in range(i + 1, n):
            if a[i] == a[j] or b[i] == b[j]:
                nt += 1
                continue
            if (a[i] < a[j] and b[i] < b[j]) or (a[i] > a[j] and b[i] > b[j]):
                nc += 1
            else:
                nd += 1
    if nc + nd == 0:
        raise ValueError("no untied pairs - gamma undefined.")
    g = float((nc - nd) / (nc + nd))
    return RichResult(
        title="Goodman-Kruskal gamma",
        summary_lines=[
            ("gamma", g),
            ("Concordant pairs", nc),
            ("Discordant pairs", nd),
            ("Tied pairs", nt),
            ("Total pairs", n * (n - 1) // 2),
            ("Strength", "negligible" if abs(g) < 0.1 else
                         "weak" if abs(g) < 0.3 else
                         "moderate" if abs(g) < 0.5 else "strong"),
        ],
        warnings=[] if nt < (nc + nd) else
                 [f"high tie rate: {nt} tied vs {nc+nd} untied; gamma robust to "
                  "ties but precision low. Consider Kendall tau-b (`kentau`)."],
        interpretation=(f"gamma={g:+.3f}; positive = concordant, negative = discordant. "
                        "Range -1 to 1. Ties dropped from numerator AND denominator."),
        payload={"value": g, "statistic": g, "concordant": nc,
                 "discordant": nd, "tied": nt},
    )

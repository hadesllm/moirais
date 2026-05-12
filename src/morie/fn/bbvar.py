# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Blackbox variance explained per dimension."""

from __future__ import annotations

from ._containers import DescriptiveResult


def bb_variance_explained(eigenvalues) -> DescriptiveResult:
    """Percent variance explained per dimension from eigenvalues.

    :param eigenvalues: Array of eigenvalues (sorted descending).
    :return: DescriptiveResult with variance proportions.

    .. epigraph:: "Ora ora ora!" -- Jotaro Kujo, JoJo's Bizarre Adventure
    """
    import numpy as np

    ev = np.asarray(eigenvalues, dtype=float).ravel()
    total = ev.sum()
    pct = (ev / total * 100).tolist() if total > 0 else [0.0] * len(ev)
    cum = np.cumsum(pct).tolist()
    return DescriptiveResult(
        name="bb_variance_explained",
        value=float(pct[0]) if pct else 0.0,
        extra={"pct_variance": pct, "cumulative": cum, "total": float(total)},
    )


bbvar = bb_variance_explained


def cheatsheet() -> str:
    return "bb_variance_explained({}) -> Blackbox variance explained per dimension."

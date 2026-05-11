# morie.fn — function file (hadesllm/morie)
"""Plot coordinates by party."""

from __future__ import annotations

from ._containers import DescriptiveResult


def plot_by_party(X, party_labels) -> DescriptiveResult:
    """Group ideal point coordinates by party for plotting.

    .. epigraph:: "Heisenberg." -- Walter White, Breaking Bad
    """
    import numpy as np

    X = np.asarray(X, dtype=float)
    labels = np.asarray(party_labels)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    parties = np.unique(labels)
    groups = {}
    for p in parties:
        mask = labels == p
        groups[str(p)] = X[mask].tolist()
    return DescriptiveResult(
        name="plot_by_party",
        value=float(len(parties)),
        extra={"groups": groups, "n_parties": len(parties), "n_total": len(labels)},
    )


plpty = plot_by_party


def cheatsheet() -> str:
    return "plot_by_party({}) -> Plot coordinates by party."

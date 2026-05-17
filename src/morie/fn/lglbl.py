# morie.fn -- function file (hadesllm/morie)
"""Label legislators with named coordinates."""

from __future__ import annotations

from ._containers import DescriptiveResult


def label_legislators(X, names) -> DescriptiveResult:
    """Pair ideal point coordinates with legislator names.

    .. epigraph:: It does not matter how slowly you go as long as you do not stop. -- Confucius
    """
    import numpy as np

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    pairs = [{"name": str(names[i]), "coords": X[i].tolist()} for i in range(len(names))]
    return DescriptiveResult(
        name="label_legislators",
        value=float(len(names)),
        extra={"labeled": pairs, "n": len(names)},
    )


lglbl = label_legislators


def cheatsheet() -> str:
    return "label_legislators({}) -> Label legislators with named coordinates."

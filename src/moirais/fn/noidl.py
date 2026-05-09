# moirais.fn — function file (hadesllm/moirais)
"""Extract ideal points from NOMINATE result."""

from __future__ import annotations

from ._containers import DescriptiveResult


def nominate_ideal_extract(result) -> DescriptiveResult:
    """Clean ideal point array from a result dict or array.

    .. epigraph:: "I chose a half measure." -- Walter White, Breaking Bad
    """
    import numpy as np

    if isinstance(result, dict):
        X = np.asarray(result.get("ideal_points", result.get("X", [])), dtype=float)
    else:
        X = np.asarray(result, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return DescriptiveResult(
        name="nominate_ideal_extract",
        value=float(np.mean(X[:, 0])),
        extra={
            "ideal_points": X,
            "n_legislators": X.shape[0],
            "n_dimensions": X.shape[1],
            "dim1_mean": float(np.mean(X[:, 0])),
            "dim1_std": float(np.std(X[:, 0])),
        },
    )


noidl = nominate_ideal_extract


def cheatsheet() -> str:
    return "nominate_ideal_extract({}) -> Extract ideal points from NOMINATE result."

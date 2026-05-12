# morie.fn -- function file (hadesllm/morie)
"""Scatter plot data for ideal points."""

from __future__ import annotations

from ._containers import DescriptiveResult


def scatter_ideal_points(X, groups=None) -> DescriptiveResult:
    """Prepare scatter data for 2-D ideal point visualization.

    :param X: n x 2 ideal point matrix.
    :param groups: Optional group labels per respondent.
    :return: DescriptiveResult with coordinate arrays.

    .. epigraph:: "Saitama punch!" -- Saitama, One Punch Man
    """
    import numpy as np

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    result = {"x": X[:, 0].tolist()}
    if X.shape[1] > 1:
        result["y"] = X[:, 1].tolist()
    result["n"] = X.shape[0]
    if groups is not None:
        result["groups"] = list(groups)
    return DescriptiveResult(name="scatter_ideal_points", value=X.shape[0], extra=result)


scidl = scatter_ideal_points


def cheatsheet() -> str:
    return "scatter_ideal_points({}) -> Scatter plot data for ideal points."

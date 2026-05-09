# moirais.fn — function file (hadesllm/moirais)
"""Neighbourhood Component Feature Selection."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def ncfs_fn(X: np.ndarray, y: np.ndarray, n_features: int = 5) -> DescriptiveResult:
    """Select features using Neighbourhood Component Feature Selection.

    :param X: 2-D array (samples x features).
    :param y: 1-D class labels.
    :param n_features: Number of features to select (default 5).
    :return: DescriptiveResult with count and selected feature indices.
    """
    from moirais._decompose import ncfs_select

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    selected = ncfs_select(X, y, n_features=n_features)
    return DescriptiveResult(
        name="ncfs",
        value=len(selected),
        extra={"selected_features": selected, "n_features": n_features},
    )


ncfs = ncfs_fn


def cheatsheet() -> str:
    return "ncfs_fn({}) -> Neighbourhood Component Feature Selection."

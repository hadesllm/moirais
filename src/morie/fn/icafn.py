# morie.fn — function file (hadesllm/morie)
"""Independent Component Analysis decomposition."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def ica_fn(X: np.ndarray, n_components: int | None = None) -> DescriptiveResult:
    """Decompose signals via Independent Component Analysis.

    :param X: 2-D array (channels x samples).
    :param n_components: Number of independent components (default all).
    :return: DescriptiveResult with component count and sources/mixing matrix.
    """
    from morie._decompose import ica_decompose

    X = np.asarray(X, dtype=float)
    sources, mixing = ica_decompose(X, n_components=n_components)
    return DescriptiveResult(
        name="ica",
        value=sources.shape[0],
        extra={"sources": sources, "mixing_matrix": mixing},
    )


icafn = ica_fn


def cheatsheet() -> str:
    return "ica_fn({}) -> Independent Component Analysis decomposition."

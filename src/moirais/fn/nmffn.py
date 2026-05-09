# moirais.fn — function file (hadesllm/moirais)
"""Non-negative Matrix Factorization."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def nmf_fn(V: np.ndarray, n_components: int = 5) -> DescriptiveResult:
    """Factorize a non-negative matrix V into W and H via NMF.

    :param V: Non-negative 2-D input matrix (n_features x n_samples).
    :param n_components: Number of components (default 5).
    :return: DescriptiveResult with W and H factor matrices in extra.
    """
    from moirais._decompose import nmf

    V = np.asarray(V, dtype=float)
    W, H = nmf(V, n_components=n_components)
    return DescriptiveResult(
        name="nmf",
        value=None,
        extra={"W": W, "H": H},
    )


nmffn = nmf_fn


def cheatsheet() -> str:
    return "nmf_fn({}) -> Non-negative Matrix Factorization."

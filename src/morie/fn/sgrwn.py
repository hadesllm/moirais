"""Row-normalize a spatial weights matrix."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def row_normalize_weights(W: np.ndarray) -> SpatialResult:
    r"""Row-standardize a spatial weights matrix.

    Parameters
    ----------
    W : np.ndarray
        Raw weights matrix, shape ``(n, n)``.

    Returns
    -------
    SpatialResult
        ``statistic`` is mean row sum (should be 1 after normalization).
        ``extra`` has ``W_normalized``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 6.

    .. epigraph::

    """
    W = np.asarray(W, dtype=np.float64).copy()
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    W_norm = W / rs

    return SpatialResult(
        name="row_normalize_weights",
        statistic=float(np.mean(W_norm.sum(axis=1))),
        p_value=None,
        extra={"W_normalized": W_norm},
    )


sgrwn = row_normalize_weights


def cheatsheet() -> str:
    return "row_normalize_weights({}) -> Row-normalize a spatial weights matrix."

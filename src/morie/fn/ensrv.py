# morie.fn — function file (hadesllm/morie)
"""Ensemble variance."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "You were my brother, Anakin."


def ensemble_variance(segments, **kwargs) -> DescriptiveResult:
    """Compute the variance across ensemble segments at each sample.

    Parameters
    ----------
    segments : array-like, shape (M, N)
        M synchronized sweeps of length N.

    Returns
    -------
    DescriptiveResult
        ``value`` is the pointwise variance (ndarray of length N).
    """
    segments = np.asarray(segments, dtype=float)
    if segments.ndim == 1:
        segments = segments.reshape(1, -1)
    var = np.var(segments, axis=0, ddof=1)
    return DescriptiveResult(
        name="ensemble_variance",
        value=var,
        extra={"M": segments.shape[0], "N": segments.shape[1], "mean_var": float(np.mean(var))},
    )


ensrv = ensemble_variance


def cheatsheet() -> str:
    return "ensemble_variance({}) -> Ensemble variance."

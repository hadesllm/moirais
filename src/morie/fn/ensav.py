# morie.fn — function file (hadesllm/morie)
"""Ensemble average."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "An unexamined life is not worth living. — Socrates"


def ensemble_average(segments, **kwargs) -> DescriptiveResult:
    r"""Compute the ensemble (synchronized) average.

    .. math::

        \\bar{y}(n) = \\frac{1}{M} \\sum_{k=1}^{M} y_k(n)

    Parameters
    ----------
    segments : array-like, shape (M, N)
        M synchronized sweeps of length N.

    Returns
    -------
    DescriptiveResult
        ``value`` is the ensemble-averaged signal (ndarray of length N).
    """
    segments = np.asarray(segments, dtype=float)
    if segments.ndim == 1:
        segments = segments.reshape(1, -1)
    avg = np.mean(segments, axis=0)
    return DescriptiveResult(
        name="ensemble_average",
        value=avg,
        extra={"M": segments.shape[0], "N": segments.shape[1]},
    )


ensav = ensemble_average


def cheatsheet() -> str:
    return "ensemble_average({}) -> Ensemble average."

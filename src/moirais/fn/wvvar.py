"""Wavelet variance (scale-dependent variance)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The Resistance will not be intimidated."


def wavelet_variance(
    x: np.ndarray,
    wavelet: str = "db4",
    level: int | None = None,
) -> DescriptiveResult:
    """Wavelet variance: variance decomposed across scales.

    .. math::

        \\sigma^2_j = \\frac{1}{N_j} \\sum_k d_j^2(k)

    Parameters
    ----------
    x : array-like
        1-D input signal.
    wavelet : str
        Wavelet name (default 'db4').
    level : int or None
        Decomposition level. Auto if None.

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``variances`` (per scale), ``total_variance``,
        ``relative_variances``.
    """
    from .dwtfn import dwt_decompose

    x = np.asarray(x, dtype=float).ravel()
    res = dwt_decompose(x, wavelet=wavelet, level=level)
    coeffs = res.extra["coeffs"]
    variances = []
    for c in coeffs:
        c = np.asarray(c, dtype=float)
        variances.append(float(np.var(c)))
    total = sum(variances)
    relative = [v / total if total > 0 else 0.0 for v in variances]
    return DescriptiveResult(
        name="wavelet_variance",
        value=float(total),
        extra={
            "variances": variances,
            "total_variance": total,
            "relative_variances": relative,
        },
    )


wvvar = wavelet_variance


def cheatsheet() -> str:
    return "wavelet_variance({}) -> Wavelet variance (scale-dependent variance)."

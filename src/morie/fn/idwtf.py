# morie.fn — function file (hadesllm/morie)
"""Inverse Discrete Wavelet Transform reconstruction."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "You underestimate my power."


def _wavelet_filter(wavelet: str = "db4"):
    """Return (lo_r, hi_r) reconstruction filters."""
    from .dwtfn import _daubechies_filter

    if wavelet in ("haar", "db1"):
        lo_d = _daubechies_filter(1)
    elif wavelet == "db2":
        lo_d = _daubechies_filter(2)
    elif wavelet == "db3":
        lo_d = _daubechies_filter(3)
    elif wavelet == "db4":
        lo_d = _daubechies_filter(4)
    else:
        raise ValueError(f"Unsupported wavelet '{wavelet}'; use haar/db1-db4")
    hi_d = np.array([(-1) ** k * lo_d[len(lo_d) - 1 - k] for k in range(len(lo_d))])
    lo_r = lo_d[::-1]
    hi_r = hi_d[::-1]
    return lo_r, hi_r


def idwt_reconstruct(
    coeffs: list,
    wavelet: str = "db4",
) -> DescriptiveResult:
    """Inverse DWT: reconstruct signal from wavelet coefficients.

    Parameters
    ----------
    coeffs : list
        Coefficient list [cA_n, cD_n, ..., cD_1] from ``dwt_decompose``.
    wavelet : str
        Wavelet name matching decomposition (default 'db4').

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``signal`` (reconstructed array).
    """
    lo_r, hi_r = _wavelet_filter(wavelet)
    approx = np.asarray(coeffs[0], dtype=float)
    for i in range(1, len(coeffs)):
        detail = np.asarray(coeffs[i], dtype=float)
        up_a = np.zeros(2 * len(approx))
        up_a[::2] = approx
        up_d = np.zeros(2 * len(detail))
        up_d[::2] = detail
        r_a = np.convolve(up_a, lo_r, mode="full")
        r_d = np.convolve(up_d, hi_r, mode="full")
        n = min(len(r_a), len(r_d))
        approx = r_a[:n] + r_d[:n]
    return DescriptiveResult(
        name="idwt_reconstruct",
        value=float(len(approx)),
        extra={"signal": approx},
    )


idwtf = idwt_reconstruct


def cheatsheet() -> str:
    return "_wavelet_filter({}) -> Inverse Discrete Wavelet Transform reconstruction."

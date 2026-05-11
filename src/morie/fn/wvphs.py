"""Phase from complex wavelet coefficients."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Twice the pride, double the fall."


def wavelet_phase(coeffs: np.ndarray) -> DescriptiveResult:
    """Extract phase from complex-valued wavelet coefficients.

    Parameters
    ----------
    coeffs : array-like
        Complex wavelet coefficients (1-D or 2-D).

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``phase`` (unwrapped), ``phase_wrapped``.
    """
    coeffs = np.asarray(coeffs, dtype=complex)
    phase_wrapped = np.angle(coeffs)
    phase = np.unwrap(phase_wrapped, axis=-1)
    return DescriptiveResult(
        name="wavelet_phase",
        value=float(np.mean(np.abs(phase))),
        extra={"phase": phase, "phase_wrapped": phase_wrapped},
    )


wvphs = wavelet_phase


def cheatsheet() -> str:
    return "wavelet_phase({}) -> Phase from complex wavelet coefficients."

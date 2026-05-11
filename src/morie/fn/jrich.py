# morie.fn — function file (hadesllm/morie)
"""Formant extraction from speech signal. 'Actions speak louder than words.' -- Jericho"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def formant_extract(
    signal: np.ndarray,
    *,
    sr: int = 16000,
    order: int = 12,
    n_formants: int = 4,
    preemph: float = 0.97,
) -> DescriptiveResult:
    """Extract formant frequencies from a speech signal via LPC analysis.

    Uses linear predictive coding (LPC) to fit an all-pole model, then finds
    formant frequencies from the roots of the LPC polynomial.

    Parameters
    ----------
    signal : array
        1-D audio signal.
    sr : int
        Sampling rate in Hz.
    order : int
        LPC order (number of poles).
    n_formants : int
        Number of formants to return.
    preemph : float
        Pre-emphasis coefficient.

    Returns
    -------
    DescriptiveResult
        ``value`` = list of formant frequencies in Hz.
    """
    sig = np.asarray(signal, dtype=float).ravel()
    if len(sig) < order + 1:
        raise ValueError(f"Signal too short for LPC order {order}")
    sig = np.append(sig[0], sig[1:] - preemph * sig[:-1])
    sig = sig * np.hamming(len(sig))
    r = np.correlate(sig, sig, mode="full")
    r = r[len(sig) - 1 :]
    R = np.zeros((order, order))
    for i in range(order):
        for j in range(order):
            R[i, j] = r[abs(i - j)]
    rhs = r[1 : order + 1]
    try:
        a = np.linalg.solve(R, rhs)
    except np.linalg.LinAlgError:
        raise ValueError("Singular autocorrelation matrix; signal may be silence")
    poly = np.concatenate([[1], -a])
    roots = np.roots(poly)
    roots = roots[np.imag(roots) >= 0]
    angles = np.angle(roots)
    freqs = np.sort(angles * sr / (2 * np.pi))
    freqs = freqs[freqs > 50]
    freqs = freqs[:n_formants]
    bandwidths = []
    for root in roots:
        if np.imag(root) >= 0 and np.angle(root) * sr / (2 * np.pi) > 50:
            bw = -sr / (2 * np.pi) * np.log(np.abs(root))
            bandwidths.append(float(bw))
    bandwidths = bandwidths[:n_formants]
    return DescriptiveResult(
        name="Formant extraction (LPC)",
        value=freqs.tolist(),
        extra={
            "sr": sr,
            "order": order,
            "n_formants_found": len(freqs),
            "bandwidths_hz": bandwidths,
            "lpc_coefficients": a.tolist(),
        },
    )


jrich = formant_extract


def cheatsheet() -> str:
    return "formant_extract({}) -> Formant extraction from speech signal. 'Actions speak louder"

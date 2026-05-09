# moirais.fn — function file (hadesllm/moirais)
"""Hilbert-Huang Transform (EMD + instantaneous frequency/amplitude).

Distinct from hhtfn.py: this implementation exposes the full Hilbert
spectrum matrix and marginal spectrum for time-frequency analysis.

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 6.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import hilbert

from ._containers import DescriptiveResult
def _sift_imf(x: np.ndarray, max_iter: int = 300, sd_tol: float = 0.05) -> np.ndarray:
    """Extract one IMF via sifting."""
    h = x.copy()
    for _ in range(max_iter):
        t = np.arange(len(h))
        max_idx = np.where((h[1:-1] > h[:-2]) & (h[1:-1] > h[2:]))[0] + 1
        min_idx = np.where((h[1:-1] < h[:-2]) & (h[1:-1] < h[2:]))[0] + 1
        if len(max_idx) < 2 or len(min_idx) < 2:
            break
        upper = CubicSpline(max_idx, h[max_idx], extrapolate=True)(t)
        lower = CubicSpline(min_idx, h[min_idx], extrapolate=True)(t)
        mean_env = (upper + lower) / 2
        prev = h.copy()
        h = h - mean_env
        sd = np.sum((prev - h) ** 2) / (np.sum(prev ** 2) + 1e-12)
        if sd < sd_tol:
            break
    return h


def hilbert_huang_spectrum(
    x: np.ndarray,
    fs: float = 1.0,
    *,
    max_imfs: int = 10,
    n_freq_bins: int = 256,
) -> DescriptiveResult:
    """Hilbert-Huang Transform with full Hilbert spectrum.

    Performs Empirical Mode Decomposition to extract IMFs, then applies
    the Hilbert transform to each IMF to obtain instantaneous
    frequency and amplitude.  Constructs a Hilbert spectrum
    :math:`H(f, t)` and marginal spectrum :math:`h(f)`:

    .. math::

        H(\\omega, t) = \\sum_{i=1}^{N} a_i(t) \\,
        \\delta(\\omega - \\omega_i(t))

    .. math::

        h(\\omega) = \\int_0^T H(\\omega, t) \\, dt

    Parameters
    ----------
    x : array-like
        1-D input signal.
    fs : float
        Sampling frequency in Hz (default 1.0).
    max_imfs : int
        Maximum number of IMFs to extract (default 10).
    n_freq_bins : int
        Number of frequency bins for the Hilbert spectrum (default 256).

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``imfs``, ``residue``,
        ``inst_freqs`` (list), ``inst_amps`` (list),
        ``hilbert_spectrum`` (n_freq_bins x n_samples),
        ``marginal_spectrum``, ``freq_axis``.

    References
    ----------
    Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
    Analysis*, 3rd ed. IEEE/Wiley, Chapter 6.

    Huang, N.E. et al. (1998). The empirical mode decomposition and
    the Hilbert spectrum for nonlinear and non-stationary time series
    analysis. *Proc. R. Soc. Lond. A*, 454, 903--995.
    doi:10.1098/rspa.1998.0193
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    residue = x.copy()
    imfs = []

    for _ in range(max_imfs):
        imf = _sift_imf(residue)
        if np.max(np.abs(imf)) < 1e-10:
            break
        imfs.append(imf)
        residue = residue - imf
        max_idx = np.where(
            (residue[1:-1] > residue[:-2]) & (residue[1:-1] > residue[2:])
        )[0] + 1
        min_idx = np.where(
            (residue[1:-1] < residue[:-2]) & (residue[1:-1] < residue[2:])
        )[0] + 1
        if len(max_idx) < 2 or len(min_idx) < 2:
            break

    inst_freqs = []
    inst_amps = []
    for imf in imfs:
        analytic = hilbert(imf)
        amp = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        freq = np.gradient(phase) * fs / (2 * np.pi)
        inst_freqs.append(freq)
        inst_amps.append(amp)

    freq_axis = np.linspace(0, fs / 2, n_freq_bins)
    hs = np.zeros((n_freq_bins, n))
    df = freq_axis[1] - freq_axis[0] if n_freq_bins > 1 else 1.0
    for freq_arr, amp_arr in zip(inst_freqs, inst_amps):
        for t_idx in range(n):
            f_val = abs(freq_arr[t_idx])
            bin_idx = int(f_val / (df + 1e-12))
            if 0 <= bin_idx < n_freq_bins:
                hs[bin_idx, t_idx] += amp_arr[t_idx] ** 2

    marginal = np.sum(hs, axis=1)

    return DescriptiveResult(
        name="hilbert_huang_spectrum",
        value=float(len(imfs)),
        extra={
            "imfs": imfs,
            "residue": residue,
            "inst_freqs": inst_freqs,
            "inst_amps": inst_amps,
            "hilbert_spectrum": hs,
            "marginal_spectrum": marginal,
            "freq_axis": freq_axis,
        },
    )


hhtrf = hilbert_huang_spectrum


def cheatsheet() -> str:
    return "hilbert_huang_spectrum({}) -> HHT with full Hilbert spectrum and marginal."

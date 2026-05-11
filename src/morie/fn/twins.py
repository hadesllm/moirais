"""Phase locking value (PLV). 'We are not here because we are free.' -- The Twins"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def phase_locking_value(
    x: np.ndarray | list,
    y: np.ndarray | list,
    *,
    fs: float = 1.0,
    band: tuple[float, float] | None = None,
) -> DescriptiveResult:
    """Phase Locking Value between two real-valued signals.

    Computes instantaneous phase via the Hilbert transform, then measures
    inter-trial or inter-signal phase consistency.

    PLV = |mean(exp(j * (phi_x - phi_y)))| in [0, 1].
    PLV = 1 means perfect phase synchronisation; PLV = 0 means none.

    Parameters
    ----------
    x, y : array-like
        Two signals of equal length.
    fs : float
        Sampling frequency (used only when *band* filtering is requested).
    band : tuple of (low, high) or None
        If provided, bandpass-filter both signals before computing PLV
        using a simple FFT rectangle filter.

    Returns
    -------
    DescriptiveResult
        ``value`` is the PLV; ``extra`` has mean phase difference and n.

    References
    ----------
    Lachaux, J.-P. et al. (1999). Measuring phase synchrony in brain signals.
    Human Brain Mapping, 8(4), 194-208.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError("x and y must be 1-D")
    if len(x_arr) != len(y_arr):
        raise ValueError("x and y must have equal length")
    n = len(x_arr)
    if n < 4:
        raise ValueError("Need at least 4 samples")

    if band is not None:
        lo, hi = band
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mask = (freqs >= lo) & (freqs <= hi)
        for arr_ref in ["x_arr", "y_arr"]:
            sig = x_arr if arr_ref == "x_arr" else y_arr
            spec = np.fft.rfft(sig)
            spec[~mask] = 0
            filtered = np.fft.irfft(spec, n=n)
            if arr_ref == "x_arr":
                x_arr = filtered
            else:
                y_arr = filtered

    def _hilbert_phase(s):
        N = len(s)
        S = np.fft.fft(s)
        h = np.zeros(N)
        if N % 2 == 0:
            h[0] = h[N // 2] = 1
            h[1 : N // 2] = 2
        else:
            h[0] = 1
            h[1 : (N + 1) // 2] = 2
        analytic = np.fft.ifft(S * h)
        return np.angle(analytic)

    phi_x = _hilbert_phase(x_arr)
    phi_y = _hilbert_phase(y_arr)
    dphi = phi_x - phi_y
    plv = float(np.abs(np.mean(np.exp(1j * dphi))))
    mean_dphi = float(np.angle(np.mean(np.exp(1j * dphi))))

    return DescriptiveResult(
        name="Phase Locking Value",
        value=plv,
        extra={
            "mean_phase_diff": mean_dphi,
            "n": n,
            "fs": fs,
            "band": list(band) if band else None,
        },
    )


twins = phase_locking_value


def cheatsheet() -> str:
    return "phase_locking_value({}) -> Phase locking value (PLV). 'We are not here because we are f"

# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""The art of doing mathematics consists in finding that special case which contains all the germs of generality. -- David Hilbert"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def time_stretch(
    signal: np.ndarray | list,
    *,
    factor: float = 2.0,
    window: int = 256,
    hop: int | None = None,
) -> DescriptiveResult:
    """Phase vocoder time-stretching (no pitch change).

    Stretches or compresses a signal in time by *factor* using the
    overlap-add phase vocoder method.

    Parameters
    ----------
    signal : array-like
        Input signal (1-D).
    factor : float
        Stretch factor (>1 = slower, <1 = faster).
    window : int
        FFT window size (must be even).
    hop : int or None
        Analysis hop size (defaults to window // 4).

    Returns
    -------
    DescriptiveResult
        ``value`` is the output length; ``extra`` has the stretched signal
        and the input/output lengths.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1 or len(x) < window:
        raise ValueError("signal must be 1-D and at least window-length")
    if factor <= 0:
        raise ValueError("factor must be > 0")
    if window % 2 != 0:
        raise ValueError("window must be even")
    if hop is None:
        hop = window // 4

    syn_hop = int(round(hop * factor))
    n_frames = (len(x) - window) // hop + 1
    out_len = (n_frames - 1) * syn_hop + window
    output = np.zeros(out_len)
    win = np.hanning(window)

    phase_acc = np.zeros(window // 2 + 1)
    prev_phase = np.zeros(window // 2 + 1)

    for i in range(n_frames):
        start = i * hop
        frame = x[start : start + window] * win
        spec = np.fft.rfft(frame)
        mag = np.abs(spec)
        phase = np.angle(spec)

        dp = phase - prev_phase
        prev_phase = phase.copy()

        dp -= hop * 2 * np.pi * np.arange(len(dp)) / window
        dp = dp - 2 * np.pi * np.round(dp / (2 * np.pi))

        freq = 2 * np.pi * np.arange(len(dp)) / window + dp / hop
        phase_acc += syn_hop * freq

        syn_spec = mag * np.exp(1j * phase_acc)
        syn_frame = np.fft.irfft(syn_spec, n=window) * win

        out_start = i * syn_hop
        out_end = out_start + window
        if out_end <= out_len:
            output[out_start:out_end] += syn_frame

    peak = np.max(np.abs(output))
    if peak > 0:
        output /= peak

    return DescriptiveResult(
        name="Time Stretch (Phase Vocoder)",
        value=len(output),
        extra={
            "stretched_signal": output.tolist(),
            "input_length": len(x),
            "output_length": len(output),
            "factor": factor,
            "window": window,
            "n_frames": n_frames,
        },
    )


bullt = time_stretch


def cheatsheet() -> str:
    return "The art of doing mathematics consists in finding that special case which contains all the germs of generality. -- David Hilbert"

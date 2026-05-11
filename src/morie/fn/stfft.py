"""Short-time Fourier Transform. 'In a dark place we find ourselves.'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def stft(
    signal: np.ndarray,
    window_size: int = 256,
    hop: int = 128,
    fs: float = 1.0,
    window: str = "hann",
) -> DescriptiveResult:
    """Compute the Short-Time Fourier Transform (STFT).

    Segments the signal into overlapping frames, applies a window
    function, and computes the FFT of each frame.

    Parameters
    ----------
    signal : ndarray, shape (n_samples,)
        Input time-domain signal.
    window_size : int
        Length of each analysis window in samples.
    hop : int
        Hop size (stride) between consecutive frames.
    fs : float
        Sampling frequency in Hz.
    window : str
        Window function name ('hann', 'hamming', 'blackman', 'rect').

    Returns
    -------
    DescriptiveResult
        name='STFT', value=n_frames,
        extra has 'magnitude' (ndarray, n_freq x n_frames),
        'phase' (ndarray), 'freq_axis' (Hz), 'time_axis' (seconds),
        'window_size', 'hop'.

    References
    ----------
    Allen, J.B. (1977). Short term spectral analysis, synthesis, and
    modification by discrete Fourier transform. *IEEE Transactions on
    Acoustics, Speech, and Signal Processing*, 25(3), 235-238.
    doi:10.1109/TASSP.1977.1162950
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    N = len(x)

    win_funcs = {
        "hann": np.hanning,
        "hamming": np.hamming,
        "blackman": np.blackman,
        "rect": lambda m: np.ones(m),
    }
    if window not in win_funcs:
        raise ValueError(f"Unknown window '{window}'; supported: {list(win_funcs)}")

    w = win_funcs[window](window_size)

    n_frames = max(0, (N - window_size) // hop + 1)
    n_freq = window_size // 2 + 1

    magnitude = np.zeros((n_freq, n_frames))
    phase = np.zeros((n_freq, n_frames))

    for i in range(n_frames):
        start = i * hop
        frame = x[start : start + window_size] * w
        X = np.fft.rfft(frame)
        magnitude[:, i] = np.abs(X)
        phase[:, i] = np.angle(X)

    freq_axis = np.fft.rfftfreq(window_size, d=1.0 / fs)
    time_axis = np.arange(n_frames) * hop / fs

    return DescriptiveResult(
        name="STFT",
        value=n_frames,
        extra={
            "magnitude": magnitude,
            "phase": phase,
            "freq_axis": freq_axis,
            "time_axis": time_axis,
            "window_size": window_size,
            "hop": hop,
            "fs": fs,
            "window": window,
        },
    )


def cheatsheet() -> str:
    return "stft({}) -> Short-time Fourier Transform. 'In a dark place we find ourse"

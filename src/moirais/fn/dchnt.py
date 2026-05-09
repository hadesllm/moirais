# moirais.fn — function file (hadesllm/moirais)
"""Ghost signal detection in noise. 'I'm already dead.' -- Deadman"""

from __future__ import annotations

import numpy as np

from ._containers import TestResult


def ghost_signal(
    data: np.ndarray | list[float],
    *,
    n_surrogates: int = 999,
    alpha: float = 0.05,
    seed: int | None = None,
) -> TestResult:
    """Detect hidden (ghost) periodic signals buried in noise via surrogate testing.

    Computes the power spectrum of the data and compares the peak spectral
    power against a null distribution generated from phase-randomized surrogates.

    Parameters
    ----------
    data : array
        1-D time series.
    n_surrogates : int
        Number of surrogates for the null distribution.
    alpha : float
        Significance level.
    seed : int or None
        Random seed.

    Returns
    -------
    TestResult
        Statistic = observed peak spectral power.
    """
    x = np.asarray(data, dtype=float).ravel()
    n = len(x)
    if n < 8:
        raise ValueError("Need at least 8 data points")
    fft_x = np.fft.rfft(x)
    power = np.abs(fft_x) ** 2
    obs_peak = float(np.max(power[1:]))
    peak_freq_idx = int(np.argmax(power[1:]) + 1)
    peak_freq = float(peak_freq_idx / n)
    rng = np.random.default_rng(seed)
    null_peaks = np.empty(n_surrogates)
    amplitudes = np.abs(fft_x)
    for i in range(n_surrogates):
        phases = rng.uniform(0, 2 * np.pi, len(fft_x))
        phases[0] = 0
        if n % 2 == 0:
            phases[-1] = 0
        surrogate_fft = amplitudes * np.exp(1j * phases)
        surrogate = np.fft.irfft(surrogate_fft, n=n)
        s_fft = np.fft.rfft(surrogate)
        s_power = np.abs(s_fft) ** 2
        null_peaks[i] = np.max(s_power[1:])
    p_value = float((null_peaks >= obs_peak).sum() + 1) / (n_surrogates + 1)
    return TestResult(
        test_name="Ghost signal detection (surrogate)",
        statistic=obs_peak,
        p_value=p_value,
        method="Phase-randomized surrogate",
        n=n,
        extra={
            "peak_frequency": peak_freq,
            "peak_period": 1.0 / peak_freq if peak_freq > 0 else float("inf"),
            "n_surrogates": n_surrogates,
            "null_95th": float(np.percentile(null_peaks, 95)),
            "signal_detected": p_value < alpha,
        },
    )


dchnt = ghost_signal


def cheatsheet() -> str:
    return "ghost_signal({}) -> Ghost signal detection in noise. 'I'm already dead.' -- Dead"

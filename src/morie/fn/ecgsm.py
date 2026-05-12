# morie.fn -- function file (hadesllm/morie)
"""Simulated 12-lead ECG generator."""

from __future__ import annotations

from ._containers import SignalResult


def ecg_12lead_simulate_fn(
    hr: float = 72.0,
    duration: float = 5.0,
    fs: float = 500.0,
) -> SignalResult:
    """Simulate a synthetic 12-lead ECG signal.

    Generates P-QRS-T morphology using Gaussian kernels with
    lead-specific gain factors.

    Parameters
    ----------
    hr : float
        Heart rate in beats per minute (default 72).
    duration : float
        Signal duration in seconds (default 5).
    fs : float
        Sampling frequency in Hz (default 500).

    Returns
    -------
    SignalResult
        *filtered* is the ``(12, n_samples)`` signal array; *extra*
        contains ``hr``, ``duration``, ``n_leads``.

    References
    ----------
    McSharry, P. E. et al. (2003). A dynamical model for generating
        synthetic electrocardiogram signals. *IEEE Trans. Biomed. Eng.*,
        50(3), 289--294.
    """
    from morie._bioplot import ecg_12lead_simulate

    signals = ecg_12lead_simulate(hr=hr, duration=duration, fs=fs)
    return SignalResult(
        name="ecg_12lead_simulate",
        filtered=signals,
        fs=fs,
        n_samples=signals.shape[1],
        extra={"hr": hr, "duration": duration, "n_leads": 12},
    )


ecgsm = ecg_12lead_simulate_fn


def cheatsheet() -> str:
    return "ecg_12lead_simulate_fn({}) -> Simulated 12-lead ECG generator."

# morie.fn -- function file (hadesllm/morie)
"""EEG band power (δ θ α β γ) via Welch -- Rangayyan Ch 9."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_eeg_bands"]


_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 100.0),
}


def rangayyan_eeg_bands(x, fs, bands=None, nperseg=None):
    """Absolute and relative band power in canonical EEG bands.

    P_band = ∫ S(f) df via Welch's PSD over band (lo, hi].

    Parameters
    ----------
    x : array-like
    fs : float
    bands : dict, optional
    nperseg : int, optional

    Returns
    -------
    RichResult with keys ``absolute`` (band->W), ``relative`` (band->fraction),
    ``total_power``, ``freqs``, ``psd``.

    References
    ----------
    Rangayyan Ch 9.
    """
    from scipy.signal import welch

    x = np.asarray(x, dtype=float)
    if nperseg is None:
        nperseg = max(16, min(x.size, int(4 * fs)))
    bands = bands or _BANDS
    freqs, pxx = welch(x, fs=fs, nperseg=nperseg)
    df = float(freqs[1] - freqs[0])
    total = float(np.trapz(pxx, freqs))
    absolute = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        absolute[name] = (float(np.trapz(pxx[mask], freqs[mask]))
                          if mask.any() else 0.0)
    relative = {k: (v / total if total > 0 else 0.0) for k, v in absolute.items()}
    rows = [[name, f"{absolute[name]:.4g}", f"{relative[name] * 100:.2f}%"]
            for name in bands]
    res = RichResult(
        title="EEG band power",
        summary_lines=[("Fs (Hz)", float(fs)),
                       ("Total power", total),
                       ("Bin width (Hz)", df)],
        tables=[{"title": "Power by band",
                 "headers": ["Band", "Absolute (W)", "Relative"],
                 "rows": rows}],
        interpretation="Relative percentages sum ≤100% (residual outside defined bands).",
        payload={"absolute": absolute, "relative": relative,
                 "total_power": total, "freqs": freqs, "psd": pxx},
    )
    return with_describe_pointer(res, "rgeeg")


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> fs = 256.0
# >>> t = np.arange(2048)/fs
# >>> x = np.sin(2*np.pi*10*t) + 0.3*rng.standard_normal(t.size)
# >>> r = rangayyan_eeg_bands(x, fs=fs)
# >>> r["absolute"]["alpha"] > r["absolute"]["gamma"]
# True


def cheatsheet():
    return "rgeeg: EEG δ θ α β γ band power -- Rangayyan Ch 9"

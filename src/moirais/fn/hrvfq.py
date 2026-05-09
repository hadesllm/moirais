# moirais.fn — function file (hadesllm/moirais)
"""Heart rate variability -- frequency-domain metrics.

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 11.
"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

__all__ = ['hrvfq']
def hrvfq(
    rr_intervals: np.ndarray,
    *,
    fs_resample: float = 4.0,
    vlf_band: tuple[float, float] = (0.003, 0.04),
    lf_band: tuple[float, float] = (0.04, 0.15),
    hf_band: tuple[float, float] = (0.15, 0.4),
    nperseg: int = 256,
) -> DescriptiveResult:
    """Frequency-domain HRV analysis (VLF, LF, HF bands).

    Parameters
    ----------
    rr_intervals : array-like
        RR intervals in seconds.
    fs_resample : float
        Resampling rate for evenly-spaced tachogram (Hz).
    vlf_band : tuple
        VLF frequency range (Hz).
    lf_band : tuple
        LF frequency range (Hz).
    hf_band : tuple
        HF frequency range (Hz).
    nperseg : int
        Welch segment length.

    Returns
    -------
    DescriptiveResult
    """
    from scipy.signal import welch

    rr = np.asarray(rr_intervals, dtype=float).ravel()
    if len(rr) < 4:
        raise ValueError("Need at least 4 RR intervals.")

    cum_t = np.cumsum(rr)
    cum_t = cum_t - cum_t[0]
    t_even = np.arange(0, cum_t[-1], 1.0 / fs_resample)
    rr_even = np.interp(t_even, cum_t, rr)
    rr_even -= np.mean(rr_even)

    seg = min(nperseg, len(rr_even))
    f, psd = welch(rr_even, fs=fs_resample, nperseg=seg)

    def band_power(lo, hi):
        mask = (f >= lo) & (f < hi)
        return float(np.trapezoid(psd[mask], f[mask])) if np.any(mask) else 0.0

    vlf = band_power(*vlf_band)
    lf = band_power(*lf_band)
    hf = band_power(*hf_band)
    total = vlf + lf + hf
    lf_hf = lf / hf if hf > 0 else float("inf")

    return DescriptiveResult(
        name="hrvfq",
        value=lf_hf,
        extra={
            "vlf": vlf,
            "lf": lf,
            "hf": hf,
            "total_power": total,
            "lf_hf_ratio": lf_hf,
            "lf_nu": lf / (lf + hf) * 100 if (lf + hf) > 0 else 0.0,
            "hf_nu": hf / (lf + hf) * 100 if (lf + hf) > 0 else 0.0,
            "frequencies": f,
            "psd": psd,
        },
    )


def cheatsheet() -> str:
    return "hrvfq({}) -> HRV frequency-domain metrics."

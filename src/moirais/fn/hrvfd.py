# moirais.fn — function file (hadesllm/moirais)
"""HRV frequency-domain metrics."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def hrv_freq_domain(
    rr: np.ndarray,
    *,
    fs_interp: float = 4.0,
) -> DescriptiveResult:
    """HRV frequency-domain analysis (VLF, LF, HF, LF/HF ratio).

    :param rr: 1-D array of RR intervals in milliseconds.
    :param fs_interp: Interpolation frequency for uniform resampling (default 4 Hz).
    :return: DescriptiveResult with power bands in ``extra``.
    """
    from scipy.signal import welch

    rr = np.asarray(rr, dtype=float).ravel()
    if len(rr) < 10:
        return DescriptiveResult(name="hrv_freq_domain", value=float("nan"))

    t_rr = np.cumsum(rr) / 1000.0
    t_rr -= t_rr[0]
    t_uniform = np.arange(0, t_rr[-1], 1.0 / fs_interp)
    rr_interp = np.interp(t_uniform, t_rr, rr)
    rr_interp -= np.mean(rr_interp)

    nperseg = min(len(rr_interp), 256)
    freqs, psd = welch(rr_interp, fs=fs_interp, nperseg=nperseg)

    vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.40)

    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    vlf = float(np.sum(psd[vlf_mask]) * df)
    lf = float(np.sum(psd[lf_mask]) * df)
    hf = float(np.sum(psd[hf_mask]) * df)
    lf_hf = lf / hf if hf > 0 else float("nan")
    total = vlf + lf + hf

    return DescriptiveResult(
        name="hrv_freq_domain",
        value=float(total),
        extra={
            "vlf": vlf,
            "lf": lf,
            "hf": hf,
            "lf_hf_ratio": lf_hf,
            "total_power": total,
            "lf_norm": lf / (lf + hf) * 100 if (lf + hf) > 0 else float("nan"),
            "hf_norm": hf / (lf + hf) * 100 if (lf + hf) > 0 else float("nan"),
        },
    )


hrvfd = hrv_freq_domain


def cheatsheet() -> str:
    return "hrv_freq_domain({}) -> HRV frequency-domain metrics."

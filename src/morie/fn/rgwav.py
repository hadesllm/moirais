# morie.fn — function file (hadesllm/morie)
"""Wavelet denoising — Rangayyan Ch 10."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_wavelet_denoise"]


def rangayyan_wavelet_denoise(x, wavelet="db4", level=None, mode="soft"):
    """Donoho-Johnstone wavelet denoising.

    Steps:

    1. DWT decomposition with ``wavelet`` to ``level`` levels.
    2. Estimate noise σ from the finest-scale detail coefficients via
       the median-absolute-deviation: σ = MAD(d1) / 0.6745.
    3. Universal threshold T = σ * sqrt(2 ln N).
    4. Apply ``soft`` (default) or ``hard`` thresholding to all detail
       coefficients (approximation untouched).
    5. Inverse DWT.

    Falls back to a moving-average smoother if ``pywt`` is unavailable
    (with a recorded warning).

    Parameters
    ----------
    x : array-like
    wavelet : str
        PyWavelets wavelet name (default ``"db4"``).
    level : int, optional
        Decomposition depth. Defaults to ``pywt.dwt_max_level``.
    mode : {"soft", "hard"}
        Thresholding rule.

    Returns
    -------
    RichResult with keys ``signal`` (denoised), ``threshold``, ``sigma``,
    ``wavelet``, ``level``, ``mode``.

    References
    ----------
    Donoho & Johnstone (1994).  Rangayyan Ch 10.
    """
    x = np.asarray(x, dtype=float).ravel()
    warnings_list: list[str] = []
    try:
        import pywt
    except ImportError:  # pragma: no cover
        warnings_list.append("pywt not available; using 5-tap moving average fallback.")
        kernel = np.ones(5) / 5.0
        y = np.convolve(x, kernel, mode="same")
        return with_describe_pointer(
            RichResult(
                title="Wavelet denoising (fallback)",
                summary_lines=[("N", x.size), ("Mode", "MA-fallback")],
                warnings=warnings_list,
                payload={"signal": y, "threshold": float("nan"),
                         "sigma": float("nan"), "wavelet": wavelet,
                         "level": 0, "mode": "MA-fallback"},
            ),
            "rgwav",
        )

    max_level = pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len)
    if level is None:
        level = max(1, min(max_level, 4))
    level = int(min(level, max_level))
    coeffs = pywt.wavedec(x, wavelet, level=level)
    # Noise estimate from finest-scale detail
    sigma = float(np.median(np.abs(coeffs[-1])) / 0.6745) if coeffs[-1].size else 0.0
    T = sigma * np.sqrt(2.0 * np.log(max(x.size, 2)))
    new_coeffs = [coeffs[0]]
    for d in coeffs[1:]:
        new_coeffs.append(pywt.threshold(d, T, mode=mode))
    y = pywt.waverec(new_coeffs, wavelet)[: x.size]
    res = RichResult(
        title="Wavelet denoising (Donoho-Johnstone)",
        summary_lines=[
            ("Wavelet", wavelet),
            ("Levels", level),
            ("σ (MAD/0.6745)", sigma),
            ("Universal threshold T", float(T)),
            ("Mode", mode),
        ],
        warnings=warnings_list,
        interpretation=(
            f"Denoised with {wavelet} at {level} levels, T={T:.4g}."
        ),
        payload={
            "signal": y,
            "threshold": float(T),
            "sigma": sigma,
            "wavelet": wavelet,
            "level": level,
            "mode": mode,
        },
    )
    return with_describe_pointer(res, "rgwav")


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> t = np.arange(100)/100.0
# >>> x = np.sin(2*np.pi*3*t) + 0.3*rng.standard_normal(100)
# >>> r = rangayyan_wavelet_denoise(x, wavelet="db4", level=3)
# >>> r["signal"].shape == x.shape
# True


def cheatsheet():
    return "rgwav: wavelet (universal-threshold) denoising — Rangayyan Ch 10"

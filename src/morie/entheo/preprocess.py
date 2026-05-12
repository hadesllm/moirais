"""
morie.entheo.preprocess — EEG-fMRI preprocessing.

Pure-NumPy fallbacks so the public API works in any CI environment;
when scipy.signal / MNE-Python / nilearn are importable we delegate
to them. Two callables:

  - preprocess_eeg : Butterworth bandpass + notch + ASR-style trimming.
  - preprocess_fmri : motion scrubbing (FD threshold) + ICA-AROMA-style
    noise-component projection (toy SVD-based stand-in in v0.4.0-alpha).

Both consume the dict-record shape emitted by ``load_dmt_imaging`` and
return a RichResult whose ``.record`` holds the cleaned record.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Tuple

import numpy as np

from ..fn._richresult import RichResult

__all__ = ["preprocess_eeg", "preprocess_fmri"]


# ---------------------------------------------------------------------------
# Filter helpers (pure-NumPy fallbacks if scipy.signal is unavailable)
# ---------------------------------------------------------------------------

def _scipy_signal():
    try:
        import scipy.signal as sps  # type: ignore
        return sps
    except Exception:
        return None


def _butter_bandpass(data: np.ndarray, sfreq: float, low: float, high: float,
                     order: int = 4) -> np.ndarray:
    sps = _scipy_signal()
    if sps is not None:
        ny = 0.5 * sfreq
        b, a = sps.butter(order, [low / ny, high / ny], btype="bandpass")
        return sps.filtfilt(b, a, data, axis=-1)
    # Fallback: FFT mask.
    n = data.shape[-1]
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    mask = (freqs >= low) & (freqs <= high)
    spec = np.fft.rfft(data, axis=-1)
    spec[..., ~mask] = 0
    return np.fft.irfft(spec, n=n, axis=-1).astype(data.dtype)


def _notch(data: np.ndarray, sfreq: float, freq: float, q: float = 30.0) -> np.ndarray:
    sps = _scipy_signal()
    if sps is not None:
        b, a = sps.iirnotch(freq, q, sfreq)
        return sps.filtfilt(b, a, data, axis=-1)
    n = data.shape[-1]
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    bw = freq / q
    notch_mask = ~((freqs >= freq - bw / 2) & (freqs <= freq + bw / 2))
    spec = np.fft.rfft(data, axis=-1)
    spec[..., ~notch_mask] = 0
    return np.fft.irfft(spec, n=n, axis=-1).astype(data.dtype)


def _asr_trim(data: np.ndarray, threshold: float) -> Tuple[np.ndarray, int]:
    """Toy ASR: clip channel-wise z-scores beyond ``threshold`` standard
    deviations and report how many samples were affected. Matches the
    spirit of EEGLAB's Artifact Subspace Reconstruction without the
    sliding-window PCA reconstruction step.
    """
    out = data.astype(np.float32, copy=True)
    mu = out.mean(axis=-1, keepdims=True)
    sd = out.std(axis=-1, keepdims=True) + 1e-9
    z = (out - mu) / sd
    bad = np.abs(z) > threshold
    n_bad = int(bad.sum())
    # Reconstruct affected samples as channel mean.
    out = np.where(bad, mu, out)
    return out, n_bad


# ---------------------------------------------------------------------------
# Public callables
# ---------------------------------------------------------------------------

def preprocess_eeg(record: dict, bandpass: Tuple[float, float] = (1.0, 40.0),
                   notch: float = 60.0, asr_threshold: float = 20.0) -> RichResult:
    """Butterworth bandpass + notch + ASR-style trimming on an EEG record.

    Parameters
    ----------
    record : dict
        Subject record from ``load_dmt_imaging()``; ``record["eeg"]``
        must contain ``sfreq`` and at least one of ``data_dmt`` /
        ``data_pcb`` as channels x timepoints arrays.
    bandpass : (low_hz, high_hz)
        Butterworth passband; defaults to 1-40 Hz.
    notch : float
        Line-noise frequency to notch (Hz). 60 (NA) or 50 (EU).
    asr_threshold : float
        Z-score above which a sample is reconstructed (toy ASR).
    """
    eeg = record.get("eeg") or {}
    sfreq = float(eeg.get("sfreq") or 250.0)
    low, high = bandpass

    cleaned = deepcopy(record)
    warnings_list: list[str] = []
    n_bad_total = 0
    n_chan = 0
    for key in ("data_dmt", "data_pcb"):
        arr = eeg.get(key)
        if arr is None:
            warnings_list.append(f"eeg.{key} absent — skipping")
            continue
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2:
            warnings_list.append(
                f"eeg.{key} ndim={arr.ndim} (expected 2); skipping")
            continue
        n_chan = max(n_chan, arr.shape[0])
        arr = _butter_bandpass(arr, sfreq, low, high)
        arr = _notch(arr, sfreq, notch)
        arr, n_bad = _asr_trim(arr, asr_threshold)
        n_bad_total += n_bad
        cleaned["eeg"][key] = arr

    return RichResult(
        title="EEG preprocessing",
        call=(f"preprocess_eeg(bandpass={bandpass}, notch={notch}, "
              f"asr_threshold={asr_threshold})"),
        summary_lines=[
            ("sfreq", sfreq),
            ("bandpass_hz", f"{low}-{high}"),
            ("notch_hz", notch),
            ("asr_threshold_z", asr_threshold),
            ("samples_reconstructed", n_bad_total),
            ("n_channels", n_chan),
        ],
        warnings=warnings_list,
        interpretation=(
            "EEG bandpass-filtered and notch-filtered; "
            f"{n_bad_total} sample(s) reconstructed by toy ASR. "
            "For production use, swap to MNE-Python (mne.io.Raw.filter "
            "and asrpy.ASR)."
        ),
        payload={"record": cleaned, "n_bad": n_bad_total,
                 "sfreq": sfreq, "bandpass": bandpass, "notch": notch},
    )


def preprocess_fmri(record: dict, motion_threshold_mm: float = 0.5,
                    n_noise_components: int = 5) -> RichResult:
    """Motion scrubbing + ICA-AROMA-style noise removal (toy SVD).

    Parameters
    ----------
    record : dict
        Subject record from ``load_dmt_imaging()``; ``record["fmri"]``
        must have ``data_dmt`` / ``data_pcb`` (parcels x timepoints)
        and optionally ``motion_fd_mm``.
    motion_threshold_mm : float
        Per-volume framewise-displacement threshold (mm). Volumes
        exceeding this are zeroed (scrubbed). Default 0.5 mm.
    n_noise_components : int
        Top-``k`` singular components projected out as a toy
        ICA-AROMA stand-in (real AROMA uses ICA + spatial classifier).
    """
    fmri = record.get("fmri") or {}
    cleaned = deepcopy(record)
    warnings_list: list[str] = []
    n_scrubbed = 0
    n_parcels = 0
    for key in ("data_dmt", "data_pcb"):
        arr = fmri.get(key)
        if arr is None:
            warnings_list.append(f"fmri.{key} absent — skipping")
            continue
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2:
            warnings_list.append(
                f"fmri.{key} ndim={arr.ndim} (expected 2); skipping")
            continue
        n_parcels = max(n_parcels, arr.shape[0])

        # 1. Motion scrubbing — zero volumes above FD threshold.
        fd = fmri.get("motion_fd_mm")
        if fd is not None:
            fd = np.asarray(fd, dtype=np.float32)
            # Pad / truncate FD to match arr.shape[-1]
            t = arr.shape[-1]
            if fd.shape[0] >= t:
                fd_t = fd[:t]
            else:
                fd_t = np.concatenate([fd, np.zeros(t - fd.shape[0], dtype=np.float32)])
            bad = fd_t > motion_threshold_mm
            n_scrubbed += int(bad.sum())
            arr = arr.copy()
            arr[:, bad] = 0.0
        else:
            warnings_list.append(
                f"fmri.motion_fd_mm absent — skipping scrubbing on {key}")

        # 2. Toy ICA-AROMA stand-in: project out top-k singular vectors.
        try:
            u, s, vt = np.linalg.svd(arr, full_matrices=False)
            k = min(n_noise_components, s.shape[0])
            # Reconstruct using all but the top-k components.
            s2 = s.copy()
            s2[:k] = 0.0
            arr = (u * s2) @ vt
        except np.linalg.LinAlgError:
            warnings_list.append(f"SVD failed on fmri.{key}; skipping AROMA")

        cleaned["fmri"][key] = arr.astype(np.float32)

    return RichResult(
        title="fMRI preprocessing",
        call=(f"preprocess_fmri(motion_threshold_mm={motion_threshold_mm}, "
              f"n_noise_components={n_noise_components})"),
        summary_lines=[
            ("motion_threshold_mm", motion_threshold_mm),
            ("volumes_scrubbed", n_scrubbed),
            ("noise_components_projected", n_noise_components),
            ("n_parcels", n_parcels),
        ],
        warnings=warnings_list,
        interpretation=(
            f"Motion-scrubbed {n_scrubbed} volume(s) above "
            f"{motion_threshold_mm} mm FD; top-{n_noise_components} "
            "singular components projected out as a toy ICA-AROMA "
            "stand-in. For production use, swap to "
            "nilearn.signal.clean + fmriprep / aroma proper."
        ),
        payload={"record": cleaned, "n_scrubbed": n_scrubbed,
                 "n_noise_components": n_noise_components},
    )

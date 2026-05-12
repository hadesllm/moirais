"""
morie.entheo._theory — Internal theoretical-framework helpers.

Per-frame phenomenal-binding metrics shared by ``analysis.py``. Kept
private because the API is unstable (v0.4.0-alpha toy implementations);
do not rely on these from outside ``morie.entheo``.

Two theories are scaffolded:

  - Beautiful Loop (Bayne, Carter, Laukkonen, Slagter)
      Phenomenal binding emerges from a predictive-coding loop that
      integrates exteroceptive (EEG-fast) and interoceptive (fMRI-slow)
      streams. We approximate "binding" as a cross-modal coupling
      score: the correlation of EEG-power envelope with fMRI gradient
      dispersion, frame-by-frame.

  - Self-Aware Networks (SAN, Carlos Pirez)
      Meta-cognitive recurrence: the network's state vector predicts
      its own next state. We approximate "recurrence" as the spectral
      slope of the autocorrelation of the EEG-fMRI joint state.

These are stand-ins for the rc1 calibrated formulations.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "binding_per_frame",
    "san_recurrence_per_frame",
    "_envelope",
    "_align_timecourses",
]


def _envelope(x: np.ndarray) -> np.ndarray:
    """Cheap power envelope: rectified + smoothed."""
    x = np.asarray(x, dtype=np.float32)
    abs_x = np.abs(x)
    # Simple 5-tap moving average along the last axis.
    kernel = np.ones(5, dtype=np.float32) / 5.0
    # Use FFT convolution along last axis for any leading dims.
    out = np.empty_like(abs_x)
    for idx in np.ndindex(abs_x.shape[:-1]):
        out[idx] = np.convolve(abs_x[idx], kernel, mode="same")
    return out


def _align_timecourses(eeg: np.ndarray, fmri: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Downsample the (longer) EEG to match the (shorter) fMRI length.

    Both inputs are (channels|parcels, time). Returns mean-collapsed
    1-D timecourses of equal length.
    """
    e = np.asarray(eeg, dtype=np.float32)
    f = np.asarray(fmri, dtype=np.float32)
    if e.ndim == 2:
        e_tc = e.mean(axis=0)
    else:
        e_tc = e
    if f.ndim == 2:
        f_tc = f.mean(axis=0)
    else:
        f_tc = f
    n = min(e_tc.shape[0], f_tc.shape[0])
    if n == 0:
        return e_tc, f_tc
    # Block-average the longer one to length n.
    def _bin(x: np.ndarray) -> np.ndarray:
        if x.shape[0] == n:
            return x
        # Trim to integer-multiple, then average.
        step = x.shape[0] // n
        if step <= 1:
            return x[:n]
        trimmed = x[: step * n]
        return trimmed.reshape(n, step).mean(axis=1)

    return _bin(e_tc), _bin(f_tc)


def binding_per_frame(eeg: np.ndarray, fmri: np.ndarray) -> np.ndarray:
    """Beautiful-Loop per-frame phenomenal-binding score.

    Returns a 1-D vector of length min(T_eeg, T_fmri) giving the local
    cross-modal coupling between the EEG power envelope and the fMRI
    gradient (first derivative). Values are zero-mean, unit-variance.
    """
    e_tc, f_tc = _align_timecourses(_envelope(eeg), fmri)
    n = min(e_tc.shape[0], f_tc.shape[0])
    if n < 4:
        return np.zeros(n, dtype=np.float32)
    f_grad = np.gradient(f_tc[:n])
    # Local 7-window correlation.
    win = 7
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        a = max(0, i - win // 2)
        b = min(n, i + win // 2 + 1)
        if b - a < 3:
            continue
        x = e_tc[a:b] - e_tc[a:b].mean()
        y = f_grad[a:b] - f_grad[a:b].mean()
        denom = (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-9)
        out[i] = float((x * y).sum() / denom)
    # Standardise.
    out -= out.mean()
    sd = out.std() + 1e-9
    out /= sd
    return out


def san_recurrence_per_frame(eeg: np.ndarray, fmri: np.ndarray) -> np.ndarray:
    """SAN per-frame meta-cognitive recurrence score.

    Returns a 1-D vector giving the lag-1 self-similarity of the
    joint EEG-fMRI state vector, computed in a sliding window. Higher
    values mean the network state predicts its own next state more
    strongly (more recurrent / more self-aware).
    """
    e_tc, f_tc = _align_timecourses(eeg, fmri)
    n = min(e_tc.shape[0], f_tc.shape[0])
    if n < 4:
        return np.zeros(n, dtype=np.float32)
    # Joint state vector: stack EEG and fMRI.
    joint = np.stack([e_tc[:n], f_tc[:n]], axis=0)
    # Z-score along time.
    joint = (joint - joint.mean(axis=1, keepdims=True)) / (
        joint.std(axis=1, keepdims=True) + 1e-9)
    win = 9
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        a = max(0, i - win // 2)
        b = min(n, i + win // 2 + 1)
        if b - a < 3:
            continue
        seg = joint[:, a:b]
        # Lag-1 autocorrelation of the joint state.
        s0 = seg[:, :-1]
        s1 = seg[:, 1:]
        num = (s0 * s1).sum()
        denom = np.sqrt((s0 ** 2).sum() * (s1 ** 2).sum()) + 1e-9
        out[i] = float(num / denom)
    return out

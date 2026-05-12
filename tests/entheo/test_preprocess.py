"""Smoke tests for morie.entheo.preprocess."""
from __future__ import annotations

import os

import numpy as np
import pytest


def _record():
    from morie.entheo.data import load_dmt_imaging
    os.environ["MORIE_DMT_IMAGING_ROOT"] = "/nope/this/does/not/exist"
    try:
        res = load_dmt_imaging(subject_id="01")
    finally:
        del os.environ["MORIE_DMT_IMAGING_ROOT"]
    return res["records"][0]


def test_preprocess_eeg_smoke():
    from morie.entheo import preprocess_eeg
    rec = _record()
    res = preprocess_eeg(rec, bandpass=(1.0, 40.0), notch=60.0,
                         asr_threshold=20.0)
    assert isinstance(res, dict)
    out = res["record"]["eeg"]["data_dmt"]
    assert isinstance(out, np.ndarray)
    assert out.shape == rec["eeg"]["data_dmt"].shape


def test_butterworth_notch_roundtrip_preserves_band():
    """Inject a 10 Hz sinusoid + 60 Hz line noise; after bandpass+notch
    the 10 Hz energy should dominate."""
    from morie.entheo.preprocess import _butter_bandpass, _notch
    sfreq = 250.0
    t = np.arange(0, 4.0, 1.0 / sfreq, dtype=np.float32)
    sig = (np.sin(2 * np.pi * 10.0 * t) + np.sin(2 * np.pi * 60.0 * t)) \
        .astype(np.float32)
    x = np.tile(sig, (4, 1))
    y = _butter_bandpass(x, sfreq, 1.0, 40.0)
    y = _notch(y, sfreq, 60.0)
    spec = np.abs(np.fft.rfft(y[0]))
    freqs = np.fft.rfftfreq(y.shape[-1], d=1.0 / sfreq)
    # Energy near 10 Hz should exceed energy near 60 Hz.
    e10 = spec[(freqs > 8) & (freqs < 12)].sum()
    e60 = spec[(freqs > 58) & (freqs < 62)].sum()
    assert e10 > 10.0 * e60


def test_preprocess_fmri_smoke():
    from morie.entheo import preprocess_fmri
    rec = _record()
    res = preprocess_fmri(rec, motion_threshold_mm=0.5)
    assert isinstance(res, dict)
    out = res["record"]["fmri"]["data_dmt"]
    assert out.shape == rec["fmri"]["data_dmt"].shape
    # AROMA SVD-projection should remove some variance from the leading
    # singular components.
    in_var = float(rec["fmri"]["data_dmt"].var())
    out_var = float(out.var())
    assert out_var <= in_var + 1e-3


def test_preprocess_fmri_scrubs_high_motion_volumes():
    from morie.entheo import preprocess_fmri
    rec = _record()
    # Inject one large FD spike.
    fd = np.asarray(rec["fmri"]["motion_fd_mm"]).copy()
    fd[5] = 5.0
    rec["fmri"]["motion_fd_mm"] = fd
    res = preprocess_fmri(rec, motion_threshold_mm=0.5)
    assert res["n_scrubbed"] >= 1

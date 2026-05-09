"""Tests for spcgm -- Spectrogram."""
import numpy as np
from moirais.fn.spcgm import spcgm
from moirais.fn._containers import DescriptiveResult


def test_spcgm_basic(signal_1khz):
    x, fs = signal_1khz
    result = spcgm(x, fs)
    assert isinstance(result, DescriptiveResult)
    assert "Sxx" in result.extra
    assert "frequencies" in result.extra
    assert "times" in result.extra


def test_spcgm_shape():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1024)
    result = spcgm(x, fs=1000.0, nperseg=128)
    Sxx = result.extra["Sxx"]
    assert Sxx.ndim == 2
    assert Sxx.shape[0] == len(result.extra["frequencies"])


def test_spcgm_nonnegative():
    x = np.random.default_rng(7).standard_normal(512)
    result = spcgm(x)
    assert np.all(result.extra["Sxx"] >= 0)

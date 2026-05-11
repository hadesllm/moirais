"""Tests for psdmt -- Multitaper PSD."""
import numpy as np
from morie.fn.psdmt import psdmt
from morie.fn._containers import DescriptiveResult


def test_psdmt_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(512)
    result = psdmt(x, fs=1000.0)
    assert isinstance(result, DescriptiveResult)
    assert "psd" in result.extra
    assert "frequencies" in result.extra


def test_psdmt_detects_tone():
    fs = 1000
    t = np.arange(0, 1.0, 1 / fs)
    x = np.sin(2 * np.pi * 100 * t)
    result = psdmt(x, fs=fs)
    freqs = result.extra["frequencies"]
    psd = result.extra["psd"]
    peak = freqs[np.argmax(psd)]
    assert abs(peak - 100) < 20


def test_psdmt_n_tapers():
    x = np.random.default_rng(7).standard_normal(256)
    result = psdmt(x, nw=3.0)
    assert result.extra["n_tapers"] == 5

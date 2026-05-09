"""Tests for emgrm -- EMG RMS envelope."""
import numpy as np
from moirais.fn.emgrm import emgrm
from moirais.fn._containers import SignalResult


def test_emgrm_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000)
    result = emgrm(x, fs=1000.0)
    assert isinstance(result, SignalResult)
    assert len(result.filtered) == 1000


def test_emgrm_nonnegative():
    x = np.random.default_rng(7).standard_normal(500)
    result = emgrm(x)
    assert np.all(result.filtered >= 0)


def test_emgrm_tracks_amplitude():
    x = np.zeros(1000)
    x[200:400] = 5.0
    result = emgrm(x, fs=1000.0, window_ms=20.0)
    assert result.filtered[300] > result.filtered[0]

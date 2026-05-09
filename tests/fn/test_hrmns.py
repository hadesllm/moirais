"""Tests for hrmns -- Harmonic analysis."""
import numpy as np
from moirais.fn.hrmns import hrmns
from moirais.fn._containers import DescriptiveResult


def test_hrmns_basic():
    fs = 1000
    t = np.arange(0, 1.0, 1 / fs)
    x = np.sin(2 * np.pi * 100 * t)
    result = hrmns(x, fs)
    assert isinstance(result, DescriptiveResult)
    assert abs(result.extra["fundamental"] - 100) < 10


def test_hrmns_finds_harmonics():
    fs = 4000
    t = np.arange(0, 1.0, 1 / fs)
    x = np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 400 * t)
    result = hrmns(x, fs, n_harmonics=3)
    assert len(result.extra["harmonics"]) >= 2


def test_hrmns_empty_band():
    fs = 100
    t = np.arange(0, 1.0, 1 / fs)
    x = np.sin(2 * np.pi * 5 * t)
    result = hrmns(x, fs, min_freq=40)
    assert result.value >= 0

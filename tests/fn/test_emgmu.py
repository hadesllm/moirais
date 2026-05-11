"""Tests for emgmu -- EMG MUAP detection."""
import numpy as np
from morie.fn.emgmu import emgmu
from morie.fn._containers import DescriptiveResult


def test_emgmu_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(2000) * 0.1
    for i in range(0, 2000, 200):
        lo = max(0, i)
        hi = min(2000, i + 10)
        x[lo:hi] = 2.0
    result = emgmu(x, fs=1000.0)
    assert isinstance(result, DescriptiveResult)
    assert result.extra["n_muaps"] > 0


def test_emgmu_no_detections():
    x = np.zeros(500)
    result = emgmu(x)
    assert result.extra["n_muaps"] == 0


def test_emgmu_fields():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(1000) * 0.1
    x[100:108] = 3.0
    x[500:508] = 3.0
    result = emgmu(x, fs=1000.0)
    for m in result.extra["muaps"]:
        assert "onset" in m
        assert "peak_amp" in m
        assert "duration_ms" in m

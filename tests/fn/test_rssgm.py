"""Tests for rssgm.py - Reassigned spectrogram."""
import numpy as np
from morie.fn.rssgm import reassigned_spectrogram, rssgm


def test_reassigned_returns_result():
    rng = np.random.default_rng(42)
    x = np.sin(np.linspace(0, 8 * np.pi, 512)) + rng.standard_normal(512) * 0.1
    result = reassigned_spectrogram(x, fs=256.0, window=64, hop=32)
    assert result.name == "reassigned_spectrogram"
    assert "magnitude" in result.extra
    assert "t_reassigned" in result.extra


def test_reassigned_shapes():
    x = np.random.default_rng(42).standard_normal(256)
    result = reassigned_spectrogram(x, fs=100.0, window=32, hop=16)
    mag = result.extra["magnitude"]
    assert mag.shape[0] > 0
    assert mag.shape[1] > 0


def test_reassigned_alias():
    x = np.random.default_rng(42).standard_normal(256)
    result = rssgm(x, fs=100.0, window=32, hop=16)
    assert result.name == "reassigned_spectrogram"

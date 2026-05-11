"""Tests for sqsgm.py - Synchrosqueezed transform."""
import numpy as np
from morie.fn.sqsgm import synchrosqueezed_transform, sqsgm


def test_sst_returns_result():
    t = np.linspace(0, 1, 200)
    x = np.sin(2 * np.pi * 10 * t)
    result = synchrosqueezed_transform(x, fs=200.0)
    assert result.name == "synchrosqueezed_transform"
    assert "sst" in result.extra


def test_sst_has_frequencies():
    x = np.random.default_rng(42).standard_normal(128)
    result = synchrosqueezed_transform(x, fs=100.0)
    assert "frequencies" in result.extra
    assert len(result.extra["frequencies"]) > 0


def test_sst_alias():
    x = np.random.default_rng(42).standard_normal(100)
    result = sqsgm(x)
    assert result.name == "synchrosqueezed_transform"

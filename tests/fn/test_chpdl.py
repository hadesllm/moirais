"""Tests for chpdl -- Chirplet decomposition."""
import numpy as np
from moirais.fn.chpdl import chpdl
from moirais.fn._containers import DescriptiveResult


def test_chpdl_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(128)
    result = chpdl(x, fs=1000.0, n_atoms=2, window_len=32)
    assert isinstance(result, DescriptiveResult)
    assert "atoms" in result.extra


def test_chpdl_captures_energy():
    fs = 500
    t = np.arange(0, 0.5, 1 / fs)
    x = np.sin(2 * np.pi * 50 * t)
    result = chpdl(x, fs=fs, n_atoms=3, window_len=32)
    assert result.extra["residual_energy"] < result.extra["original_energy"]


def test_chpdl_returns_atoms():
    x = np.random.default_rng(7).standard_normal(64)
    result = chpdl(x, n_atoms=2, window_len=16)
    for atom in result.extra["atoms"]:
        assert "t0" in atom
        assert "f0" in atom
        assert "coefficient" in atom

"""Tests for mptfd.py - Matching Pursuit Time-Frequency Distribution."""
import numpy as np
import pytest
from morie.fn.mptfd import mptfd_fn, mptfd


def test_mptfd_returns_descriptive_result():
    t = np.linspace(0, 1, 128)
    x = np.sin(2 * np.pi * 10 * t)
    result = mptfd_fn(x, n_atoms=5, fs=128.0)
    assert result.name == "mp_tfd"
    assert isinstance(result.value, (int, np.integer))
    assert "tfd" in result.extra
    assert "time" in result.extra
    assert "frequencies" in result.extra


def test_mptfd_shapes():
    n = 128
    x = np.random.default_rng(42).standard_normal(n)
    result = mptfd_fn(x, n_atoms=5, fs=1.0)
    tfd = result.extra["tfd"]
    assert tfd.shape == (n // 2, n)
    assert len(result.extra["time"]) == n
    assert len(result.extra["frequencies"]) == n // 2


def test_mptfd_nonnegative_energy():
    x = np.random.default_rng(42).standard_normal(64)
    result = mptfd_fn(x, n_atoms=10, fs=1.0)
    assert np.all(result.extra["tfd"] >= 0)


def test_mptfd_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = mptfd(x, n_atoms=3)
    assert result.name == "mp_tfd"

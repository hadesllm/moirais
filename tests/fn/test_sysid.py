"""Tests for sysid.py - System identification."""
import numpy as np
from morie.fn.sysid import system_identify_fn, sysid


def test_sysid_returns_result():
    rng = np.random.default_rng(42)
    u = rng.standard_normal(200)
    h_true = np.array([1.0, 0.5, -0.3])
    y = np.convolve(u, h_true)[:200] + 0.01 * rng.standard_normal(200)
    result = system_identify_fn(u, y, order=3)
    assert result.name == "system_identify"
    assert "impulse_response" in result.extra


def test_sysid_recovers_impulse():
    rng = np.random.default_rng(42)
    u = rng.standard_normal(500)
    h_true = np.array([1.0, 0.5])
    y = np.convolve(u, h_true)[:500]
    result = system_identify_fn(u, y, order=2)
    np.testing.assert_allclose(result.extra["impulse_response"], h_true, atol=0.05)


def test_sysid_alias():
    rng = np.random.default_rng(42)
    u = rng.standard_normal(100)
    y = rng.standard_normal(100)
    result = sysid(u, y, order=2)
    assert result.name == "system_identify"

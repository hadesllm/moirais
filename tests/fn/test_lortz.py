"""Tests for moirais.fn.lortz -- Lorentz transformation."""

import numpy as np
import pytest

from moirais.fn.lortz import lortz


def test_returns_dict():
    r = lortz(np.array([1.0, 0.0, 0.0, 0.0]), v=0.0)
    assert isinstance(r, dict)
    for k in ("event_prime", "gamma", "beta", "boost_matrix"):
        assert k in r


def test_zero_velocity_identity():
    ev = np.array([3.0, 1.0, 2.0, 0.5])
    r = lortz(ev, v=0.0)
    np.testing.assert_allclose(r["event_prime"], ev, atol=1e-14)
    assert r["gamma"] == pytest.approx(1.0)
    assert r["beta"] == pytest.approx(0.0)


def test_gamma_half_c():
    c = 299792458.0
    r = lortz(np.array([c, 0, 0, 0]), v=0.5 * c)
    expected_gamma = 1.0 / np.sqrt(1.0 - 0.25)
    assert r["gamma"] == pytest.approx(expected_gamma, rel=1e-12)


def test_time_dilation():
    c = 3e8
    dt = 1.0
    r = lortz(np.array([c * dt, 0, 0, 0]), v=0.6 * c, c=c)
    gamma = 1.0 / np.sqrt(1.0 - 0.36)
    assert r["event_prime"][0] == pytest.approx(gamma * c * dt, rel=1e-10)


def test_boost_matrix_det():
    r = lortz(np.array([1, 0, 0, 0]), v=0.9 * 299792458.0)
    det = np.linalg.det(r["boost_matrix"])
    assert det == pytest.approx(1.0, abs=1e-10)


def test_exceeds_c_raises():
    with pytest.raises(ValueError, match="< c"):
        lortz(np.array([1, 0, 0, 0]), v=3e8)


def test_wrong_shape_raises():
    with pytest.raises(ValueError):
        lortz(np.array([1, 0, 0]), v=0.5)

"""Tests for morie.fn.frdeq -- Friedmann equations."""

import numpy as np
import pytest

from morie.fn.frdeq import frdeq


def test_returns_dict():
    r = frdeq()
    assert isinstance(r, dict)
    for k in ("a", "H", "Omega_k", "Omega_Lambda", "deceleration_q"):
        assert k in r


def test_flat_universe_omega_k_zero():
    r = frdeq(k=0)
    assert r["Omega_k"] == pytest.approx(0.0, abs=1e-10)


def test_h0_at_a_equals_1():
    r = frdeq(H0=70.0, a_range=(0.99, 1.01), n_points=3)
    idx = np.argmin(np.abs(r["a"] - 1.0))
    assert r["H"][idx] == pytest.approx(70.0, rel=0.01)


def test_hubble_positive():
    r = frdeq()
    assert np.all(r["H"] >= 0)


def test_negative_h0_raises():
    with pytest.raises(ValueError):
        frdeq(H0=-10)


def test_n_points():
    r = frdeq(n_points=100)
    assert len(r["a"]) == 100
    assert len(r["H"]) == 100

"""Tests for morie.fn.steng -- stress-energy tensor."""

import numpy as np
import pytest

from morie.fn.steng import steng


def test_returns_dict():
    u = np.array([1.0, 0.0, 0.0, 0.0])
    g = np.diag([-1.0, 1.0, 1.0, 1.0])
    r = steng(rho=1e10, p=1e8, u=u, metric=g)
    assert isinstance(r, dict)
    assert "stress_energy" in r
    assert "trace" in r


def test_shape():
    u = np.array([1.0, 0.0, 0.0, 0.0])
    g = np.diag([-1.0, 1.0, 1.0, 1.0])
    r = steng(rho=1e10, p=1e8, u=u, metric=g)
    assert r["stress_energy"].shape == (4, 4)


def test_dust_zero_pressure():
    u = np.array([1.0, 0.0, 0.0, 0.0])
    g = np.diag([-1.0, 1.0, 1.0, 1.0])
    rho = 1e10
    r = steng(rho=rho, p=0.0, u=u, metric=g)
    assert r["stress_energy"][0, 0] == pytest.approx(rho, rel=1e-10)


def test_wrong_shape_raises():
    with pytest.raises(ValueError):
        steng(rho=1.0, p=0.0, u=np.zeros(3), metric=np.eye(4))

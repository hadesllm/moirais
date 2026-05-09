"""Tests for moirais.fn.maxwl -- Maxwell equations."""

import numpy as np
import pytest

from moirais.fn.maxwl import maxwl


def test_returns_dict():
    E = np.array([1e3, 0, 0])
    B = np.array([0, 0, 1e-3])
    r = maxwl(E, B)
    assert isinstance(r, dict)
    for k in ("field_tensor", "dual_tensor", "invariant_1",
              "invariant_2", "energy_density", "poynting_vector"):
        assert k in r


def test_field_tensor_antisymmetric():
    E = np.array([1e3, 2e3, 3e3])
    B = np.array([1e-3, 2e-3, 3e-3])
    r = maxwl(E, B)
    F = r["field_tensor"]
    np.testing.assert_allclose(F, -F.T, atol=1e-20)


def test_pure_E_invariants():
    E = np.array([1e3, 0, 0])
    B = np.array([0, 0, 0])
    r = maxwl(E, B)
    c = 299792458.0
    assert r["invariant_1"] == pytest.approx(-E[0] ** 2 / c ** 2, rel=1e-6)
    assert r["invariant_2"] == pytest.approx(0.0, abs=1e-10)


def test_energy_density_positive():
    E = np.array([100, 200, 300])
    B = np.array([0.01, 0.02, 0.03])
    r = maxwl(E, B)
    assert r["energy_density"] > 0


def test_wrong_shape_raises():
    with pytest.raises(ValueError):
        maxwl(np.zeros(2), np.zeros(3))

"""Tests for morie.fn.klgrd -- Klein-Gordon equation."""

import numpy as np
import pytest

from morie.fn.klgrd import klgrd


def test_returns_dict():
    r = klgrd(m=1.0, n_x=50, n_t=20, t_span=(0, 1))
    assert isinstance(r, dict)
    for k in ("x", "t", "phi", "dispersion_relation"):
        assert k in r


def test_shape():
    r = klgrd(m=1.0, n_x=50, n_t=20, t_span=(0, 1))
    assert r["phi"].shape == (20, 50)
    assert len(r["x"]) == 50
    assert len(r["t"]) == 20


def test_dispersion_relation():
    r = klgrd(m=1.0, n_x=100, hbar=1.0, c=1.0)
    k = r["dispersion_relation"]["k"]
    omega = r["dispersion_relation"]["omega"]
    expected = np.sqrt(k ** 2 + 1.0)
    np.testing.assert_allclose(omega, expected, atol=1e-10)


def test_massless_limit():
    r = klgrd(m=0.0, n_x=100, hbar=1.0, c=1.0)
    k = r["dispersion_relation"]["k"]
    omega = r["dispersion_relation"]["omega"]
    np.testing.assert_allclose(omega, np.abs(k), atol=1e-10)


def test_negative_mass_raises():
    with pytest.raises(ValueError, match="Mass"):
        klgrd(m=-1.0)

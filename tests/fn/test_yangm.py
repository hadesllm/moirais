"""Tests for morie.fn.yangm -- Yang-Mills action."""

import numpy as np
import pytest

from morie.fn.yangm import yangm


def test_returns_dict():
    A = np.zeros((3, 4))
    r = yangm(A)
    assert isinstance(r, dict)
    for k in ("field_strength", "action_density", "structure_constants"):
        assert k in r


def test_zero_field_zero_action():
    A = np.zeros((3, 4))
    r = yangm(A)
    assert r["action_density"] == pytest.approx(0.0, abs=1e-14)
    np.testing.assert_allclose(r["field_strength"], 0.0, atol=1e-14)


def test_field_strength_shape():
    A = np.ones((3, 4)) * 0.1
    r = yangm(A)
    assert r["field_strength"].shape == (3, 4, 4)


def test_structure_constants_antisymmetric():
    A = np.zeros((3, 4))
    r = yangm(A)
    f = r["structure_constants"]
    assert f[0, 1, 2] == pytest.approx(-f[0, 2, 1])
    assert f[0, 1, 2] == pytest.approx(1.0)


def test_wrong_shape_raises():
    with pytest.raises(ValueError):
        yangm(np.zeros((3, 3)))

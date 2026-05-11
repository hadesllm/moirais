"""Tests for morie.fn.ricci -- Ricci tensor."""

import numpy as np
import pytest

from morie.fn.ricci import ricci


def test_returns_dict():
    R = np.zeros((4, 4, 4, 4))
    g = np.diag([-1.0, 1.0, 1.0, 1.0])
    r = ricci(R, g)
    assert isinstance(r, dict)
    for k in ("ricci_tensor", "scalar_curvature", "metric_inverse"):
        assert k in r


def test_flat_space_zero():
    R = np.zeros((4, 4, 4, 4))
    g = np.diag([-1.0, 1.0, 1.0, 1.0])
    r = ricci(R, g)
    np.testing.assert_allclose(r["ricci_tensor"], 0.0, atol=1e-14)
    assert r["scalar_curvature"] == pytest.approx(0.0, abs=1e-14)


def test_shape_mismatch_raises():
    R = np.zeros((3, 3, 3, 3))
    g = np.diag([-1.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        ricci(R, g)


def test_nonzero_riemann():
    R = np.zeros((2, 2, 2, 2))
    R[0, 1, 0, 1] = 1.0
    R[1, 0, 1, 0] = 1.0
    g = np.eye(2)
    r = ricci(R, g)
    assert r["ricci_tensor"].shape == (2, 2)
    assert r["scalar_curvature"] != 0.0

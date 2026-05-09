"""Tests for moirais.fn.chrsf -- Christoffel symbols."""

import numpy as np
import pytest

from moirais.fn.chrsf import chrsf


def test_returns_dict():
    g = np.diag([-1.0, 1.0, 1.0, 1.0])
    r = chrsf(g)
    assert isinstance(r, dict)
    assert "christoffel" in r
    assert "metric_inverse" in r


def test_flat_metric_zero_christoffel():
    g = np.diag([-1.0, 1.0, 1.0, 1.0])
    r = chrsf(g)
    np.testing.assert_allclose(r["christoffel"], 0.0, atol=1e-14)


def test_metric_inverse():
    g = np.diag([-1.0, 1.0, 1.0, 1.0])
    r = chrsf(g)
    prod = g @ r["metric_inverse"]
    np.testing.assert_allclose(prod, np.eye(4), atol=1e-14)


def test_numerical_derivative():
    def sphere_metric(coords):
        r, theta = coords[0], coords[1]
        g = np.diag([1.0, r ** 2])
        return g

    g = sphere_metric(np.array([2.0, np.pi / 4]))
    r = chrsf(g, coords=np.array([2.0, np.pi / 4]),
              metric_func=sphere_metric)
    assert r["christoffel"].shape == (2, 2, 2)


def test_non_square_raises():
    with pytest.raises(ValueError):
        chrsf(np.ones((3, 4)))

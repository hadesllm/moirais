"""Tests for moirais.fn.geods -- geodesic equation solver."""

import numpy as np
import pytest

from moirais.fn.geods import geods


def flat_metric(x):
    return np.diag([-1.0, 1.0, 1.0, 1.0])


def test_returns_dict():
    r = geods(flat_metric, np.array([0, 1, 0, 0]), np.array([1, 0.5, 0, 0]),
              tau_span=(0, 1), n_points=20)
    assert isinstance(r, dict)
    for k in ("tau", "position", "velocity"):
        assert k in r


def test_flat_space_straight_line():
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    u0 = np.array([1.0, 0.5, 0.0, 0.0])
    r = geods(flat_metric, x0, u0, tau_span=(0, 2), n_points=50)
    final_pos = r["position"][-1]
    assert final_pos[1] == pytest.approx(1.0, abs=0.05)


def test_velocity_conservation_flat():
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    u0 = np.array([1.0, 0.3, 0.2, 0.0])
    r = geods(flat_metric, x0, u0, tau_span=(0, 1), n_points=30)
    v_init = r["velocity"][0]
    v_final = r["velocity"][-1]
    np.testing.assert_allclose(v_init, v_final, atol=1e-6)


def test_wrong_shape_raises():
    with pytest.raises(ValueError):
        geods(flat_metric, np.zeros(3), np.zeros(4))

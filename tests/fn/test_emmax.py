"""Tests for morie.fn.emmax -- EM maximization step."""
import numpy as np
from morie.fn.emmax import em_maximization_step, emmax


def test_alias():
    assert emmax is em_maximization_step


def test_smoke():
    theta = np.array([0.5, -0.3, 0.1])
    r = em_maximization_step(Q=-10.0, theta=theta)
    assert r.name == "em_maximization_step"
    assert r.extra["n_params"] == 3
    assert len(r.extra["updated_theta"]) == 3

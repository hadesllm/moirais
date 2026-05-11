"""Tests for morie.fn.irthx -- heteroskedastic IRT."""
import numpy as np
from morie.fn.irthx import irt_heteroskedastic, irthx


def test_alias():
    assert irthx is irt_heteroskedastic


def test_smoke():
    theta = np.array([0.0, 1.0, -1.0])
    sigma = np.array([1.0, 1.0, 1.0])
    r = irt_heteroskedastic(theta, alpha=1.0, beta=0.0, sigma_i=sigma)
    assert r.name == "irt_heteroskedastic"
    assert len(r.extra["probabilities"]) == 3


def test_high_variance():
    theta = np.array([1.0])
    r = irt_heteroskedastic(theta, alpha=1.0, beta=0.0, sigma_i=np.array([5.0]))
    assert 0.4 < r.extra["probabilities"][0] < 0.6

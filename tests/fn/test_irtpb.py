"""Tests for morie.fn.irtpb -- IRT probability."""
import numpy as np
from morie.fn.irtpb import irt_probability, irtpb


def test_alias():
    assert irtpb is irt_probability


def test_smoke():
    theta = np.array([-2, -1, 0, 1, 2], dtype=float)
    r = irt_probability(theta, alpha=1.5, beta=0.0)
    assert r.name == "irt_probability"
    assert len(r.extra["probabilities"]) == 5
    assert r.extra["probabilities"][2] > 0.49  # at theta=beta


def test_high_ability():
    r = irt_probability([3.0], alpha=1.0, beta=0.0)
    assert r.extra["probabilities"][0] > 0.9

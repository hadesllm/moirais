"""Tests for logistic_map."""
import pytest
from morie.fn.chaos import logistic_map

class TestChaos:
    def test_fixed_point(self):
        r = logistic_map(2.0, x0=0.5, n_iter=100)
        assert r.value == pytest.approx(0.5, abs=0.01)

    def test_chaotic(self):
        r = logistic_map(3.9, x0=0.5, n_iter=200)
        assert r.extra["unique_attractors_approx"] > 10

"""Tests for moirais.fn.hamam -- Brownian motion."""

from moirais.fn.hamam import brownian_motion, hamam
from moirais.fn._containers import DescriptiveResult


class TestHamam:
    def test_alias(self):
        assert hamam is brownian_motion

    def test_standard_bm(self):
        result = brownian_motion(n_steps=500, n_paths=10, mu=0.0, sigma=1.0, seed=42)
        assert isinstance(result, DescriptiveResult)
        assert result.extra["n_paths"] == 10

    def test_geometric_bm(self):
        result = brownian_motion(n_steps=200, n_paths=5, mu=0.05, sigma=0.2, seed=42)
        assert result.extra["mu"] == 0.05
        assert result.value > 0

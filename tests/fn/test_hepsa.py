"""Tests for morie.fn.hepsa -- PSA."""

import pytest
from morie.fn.hepsa import probabilistic_sensitivity


class TestPSA:
    def test_basic(self):
        res = probabilistic_sensitivity(
            {"cost": (1000, 200), "effect": (0.5, 0.1)},
            n_sim=500,
        )
        assert res.name == "PSA"
        assert res.extra["n_params"] == 2

    def test_means_close(self):
        res = probabilistic_sensitivity({"x": (100, 10)}, n_sim=10000)
        assert abs(res.value["x"] - 100) < 5

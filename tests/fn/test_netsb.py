"""Tests for moirais.fn.netsb — Network stability (CS coefficient)."""

import numpy as np
import pytest
from moirais.fn.netsb import network_stability


class TestNetworkStability:

    def test_returns_dict(self, rng):
        X = rng.standard_normal((50, 5))
        result = network_stability(X, n_boot=20)
        assert "cs_strength" in result
        assert "cs_ei" in result

    def test_cs_in_range(self, rng):
        X = rng.standard_normal((80, 5))
        result = network_stability(X, n_boot=20)
        assert 0 <= result["cs_strength"] <= 1
        assert 0 <= result["cs_ei"] <= 1

    def test_edge_ci_shape(self, rng):
        X = rng.standard_normal((50, 4))
        result = network_stability(X, n_boot=20)
        assert result["edge_ci_lower"].shape == (4, 4)
        assert result["edge_ci_upper"].shape == (4, 4)

    def test_n_boot_recorded(self, rng):
        X = rng.standard_normal((50, 3))
        result = network_stability(X, n_boot=30)
        assert result["n_boot"] == 30

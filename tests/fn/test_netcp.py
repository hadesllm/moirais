"""Tests for moirais.fn.netcp — Network comparison test."""

import numpy as np
import pytest
from moirais.fn.netcp import network_compare


class TestNetworkCompare:

    def test_returns_dict(self, rng):
        X1 = rng.standard_normal((50, 5))
        X2 = rng.standard_normal((50, 5))
        result = network_compare(X1, X2, n_perm=20)
        assert "global_strength_p" in result
        assert "structure_p" in result

    def test_same_data_no_difference(self, rng):
        X = rng.standard_normal((50, 5))
        result = network_compare(X, X, n_perm=50)
        # p-values should be high (no difference)
        assert result["global_strength_p"] > 0.05

    def test_p_values_in_range(self, rng):
        X1 = rng.standard_normal((40, 4))
        X2 = rng.standard_normal((40, 4))
        result = network_compare(X1, X2, n_perm=30)
        assert 0 <= result["global_strength_p"] <= 1
        assert 0 <= result["structure_p"] <= 1

    def test_n_perm_recorded(self, rng):
        X1 = rng.standard_normal((30, 3))
        X2 = rng.standard_normal((30, 3))
        result = network_compare(X1, X2, n_perm=25)
        assert result["n_perm"] == 25

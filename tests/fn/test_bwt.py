"""Tests for moirais.fn.bwt -- RDD bandwidth selection."""

import numpy as np
import pytest
from moirais.fn.bwt import rdd_bandwidth


class TestRDDBandwidth:
    def test_bandwidth_positive(self):
        rng = np.random.default_rng(42)
        n = 200
        rv = rng.uniform(-5, 5, n)
        y = 1.0 * (rv >= 0) + rng.normal(0, 1, n)
        result = rdd_bandwidth(rv, y, cutoff=0.0)
        assert result["h_opt"] > 0

    def test_h_half_and_double(self):
        rng = np.random.default_rng(42)
        rv = rng.uniform(-3, 3, 100)
        y = 0.5 * (rv >= 0) + rng.normal(0, 0.5, 100)
        result = rdd_bandwidth(rv, y, cutoff=0.0)
        assert result["h_half"] == pytest.approx(result["h_opt"] / 2)
        assert result["h_double"] == pytest.approx(result["h_opt"] * 2)

    def test_counts_sides(self):
        rng = np.random.default_rng(42)
        rv = rng.uniform(-2, 2, 100)
        y = rng.normal(0, 1, 100)
        result = rdd_bandwidth(rv, y, cutoff=0.0)
        assert result["n_left"] + result["n_right"] == 100

    def test_too_few_on_side_raises(self):
        with pytest.raises(ValueError, match="5"):
            rdd_bandwidth([1, 2, 3, 4, 5, 6, 7, 8], [1]*8, cutoff=0.0)

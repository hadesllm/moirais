"""Tests for morie.fn.chrnf -- Chernoff distribution."""

import numpy as np
import pytest
from morie.fn.chrnf import chernoff_distribution


class TestChernoffDistribution:
    def test_basic_output(self):
        x = np.linspace(-2, 2, 50)
        r = chernoff_distribution(x)
        assert len(r["cdf"]) == 50
        assert len(r["pdf"]) == 50

    def test_cdf_monotone(self):
        x = np.linspace(-3, 3, 100)
        r = chernoff_distribution(x)
        cdf = np.array(r["cdf"])
        assert np.all(np.diff(cdf) >= -1e-6)

    def test_cdf_bounds(self):
        x = np.linspace(-5, 5, 100)
        r = chernoff_distribution(x, n_grid=300)
        assert r["cdf"][0] < 0.1
        assert r["cdf"][-1] > 0.9

    def test_mean_near_zero(self):
        x = np.linspace(-3, 3, 50)
        r = chernoff_distribution(x, n_grid=200)
        assert abs(r["mean"]) < 0.5

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            chernoff_distribution(np.array([]))

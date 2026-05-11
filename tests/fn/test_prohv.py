"""Tests for morie.fn.prohv -- Prohorov metric."""

import numpy as np
import pytest
from morie.fn.prohv import prohorov_metric


class TestProhorov:
    def test_identical_samples(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = prohorov_metric(x, x)
        assert r["prohorov"] < 0.01
        assert r["ks_statistic"] < 0.01

    def test_disjoint_samples(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([10.0, 11.0, 12.0])
        r = prohorov_metric(x, y)
        assert r["prohorov"] > 0.5

    def test_ks_lower_bound(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(50)
        y = rng.standard_normal(50) + 1.0
        r = prohorov_metric(x, y)
        assert r["prohorov"] >= r["ks_lower"] - 1e-8
        assert r["prohorov"] <= r["ks_statistic"] + 1e-8

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            prohorov_metric(np.array([]), np.array([1.0]))

    def test_symmetry(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(30)
        y = rng.uniform(0, 1, 30)
        r1 = prohorov_metric(x, y)["prohorov"]
        r2 = prohorov_metric(y, x)["prohorov"]
        assert r1 == pytest.approx(r2, abs=1e-6)

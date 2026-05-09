"""Tests for moirais.fn.tost — TOST equivalence test."""
import numpy as np
import pytest
from moirais.fn.tost import tost_test


class TestTOST:
    def test_equivalent_samples(self):
        rng = np.random.default_rng(42)
        x = rng.normal(10, 1, 100)
        y = rng.normal(10.1, 1, 100)
        res = tost_test(x, y, margin=0.5)
        assert res.extra["p_value"] < 0.05
        assert res.extra["margin"] == 0.5
        assert res.extra["equivalent"] is True

    def test_non_equivalent_samples(self):
        rng = np.random.default_rng(42)
        x = rng.normal(10, 1, 50)
        y = rng.normal(15, 1, 50)
        res = tost_test(x, y, margin=0.5)
        assert res.extra["p_value"] > 0.05
        assert res.extra["equivalent"] is False

    def test_margin_stored(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 80)
        y = rng.normal(0, 1, 80)
        res = tost_test(x, y, margin=1.0)
        assert res.extra["margin"] == 1.0

"""Tests for moirais.fn.dts -- Hartigan's dip test."""

import numpy as np
import pytest
from moirais.fn.dts import dip_test
from moirais.fn._containers import TestResult


class TestDip:
    def test_unimodal(self):
        """Normal data => non-significant (unimodal)."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        r = dip_test(x, n_boot=200, seed=42)
        assert isinstance(r, TestResult)
        assert r.p_value > 0.05

    def test_bimodal(self):
        """Mixture of two well-separated normals => significant."""
        rng = np.random.default_rng(42)
        x = np.concatenate([rng.normal(-5, 0.5, 100), rng.normal(5, 0.5, 100)])
        r = dip_test(x, n_boot=200, seed=42)
        assert r.p_value < 0.05

    def test_raises_small_n(self):
        with pytest.raises(ValueError):
            dip_test([1, 2, 3])

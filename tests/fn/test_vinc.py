"""Tests for morie.fn.vinc — Incremental validity."""

import numpy as np
import pytest
from morie.fn.vinc import validity_incremental


class TestValidityIncremental:

    def test_positive_delta_r2(self, rng):
        n = 200
        x_base = rng.standard_normal(n)
        x_new = rng.standard_normal(n)
        y = 0.5 * x_base + 0.8 * x_new + rng.standard_normal(n) * 0.3
        result = validity_incremental(y, x_base, x_new)
        assert result["delta_r2"] > 0

    def test_f_test_significant(self, rng):
        n = 200
        x_base = rng.standard_normal(n)
        x_new = rng.standard_normal(n)
        y = 0.5 * x_base + 1.0 * x_new + rng.standard_normal(n) * 0.3
        result = validity_incremental(y, x_base, x_new)
        assert result["p_value"] < 0.05

    def test_r2_full_ge_base(self, rng):
        n = 100
        x_base = rng.standard_normal(n)
        x_new = rng.standard_normal(n)
        y = x_base + x_new + rng.standard_normal(n) * 0.5
        result = validity_incremental(y, x_base, x_new)
        assert result["r2_full"] >= result["r2_base"] - 1e-10

    def test_n_correct(self, rng):
        n = 80
        result = validity_incremental(
            rng.standard_normal(n),
            rng.standard_normal(n),
            rng.standard_normal(n),
        )
        assert result["n"] == n

    def test_multivariate_base(self, rng):
        n = 100
        x_base = rng.standard_normal((n, 3))
        x_new = rng.standard_normal(n)
        y = x_base @ [1, 2, 3] + 2 * x_new + rng.standard_normal(n) * 0.5
        result = validity_incremental(y, x_base, x_new)
        assert result["delta_r2"] > 0

"""Tests for morie.fn.turbr — turbo code interleaver."""

import numpy as np
import pytest

from morie.fn.turbr import turbr


class TestTurbr:
    def test_permutation_valid(self):
        result = turbr(100)
        assert len(result["permutation"]) == 100
        assert len(set(result["permutation"])) == 100

    def test_inverse_correct(self):
        result = turbr(50)
        perm = result["permutation"]
        inv = result["inverse"]
        np.testing.assert_array_equal(perm[inv], np.arange(50))

    def test_qpp_method(self):
        result = turbr(64, method="qpp")
        assert len(set(result["permutation"])) == 64

    def test_golden_method(self):
        result = turbr(32, method="golden")
        assert len(result["permutation"]) == 32

    def test_spread_positive(self):
        result = turbr(100)
        assert result["spread"] >= 0.0

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            turbr(1)

    def test_unknown_method(self):
        with pytest.raises(ValueError):
            turbr(10, method="invalid")

    def test_deterministic(self):
        r1 = turbr(50, seed=123)
        r2 = turbr(50, seed=123)
        np.testing.assert_array_equal(r1["permutation"], r2["permutation"])

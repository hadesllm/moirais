"""Tests for moirais.fn.berns -- Bernstein concentration inequality."""

import pytest
from moirais.fn.berns import bernstein_bound


class TestBernstein:
    def test_basic_bound(self):
        r = bernstein_bound(100, 5.0)
        assert 0 < r["bound"] <= 1

    def test_tighter_than_hoeffding_for_small_variance(self):
        r = bernstein_bound(100, 5.0, sigma_sq=0.01)
        assert r["bound"] < 0.5

    def test_variance_and_linear_terms(self):
        r = bernstein_bound(50, 3.0, sigma_sq=0.25, M=1.0)
        assert r["variance_term"] == pytest.approx(12.5)
        assert r["linear_term"] == pytest.approx(1.0)

    def test_bound_decreases_with_t(self):
        b1 = bernstein_bound(100, 3.0)["bound"]
        b2 = bernstein_bound(100, 10.0)["bound"]
        assert b2 < b1

    def test_invalid_sigma(self):
        with pytest.raises(ValueError, match="sigma_sq"):
            bernstein_bound(100, 1.0, sigma_sq=-1)

    def test_invalid_M(self):
        with pytest.raises(ValueError, match="M must be"):
            bernstein_bound(100, 1.0, M=0)

"""Tests for morie.fn.splhf — Spearman-Brown split-half reliability."""

import pytest
import numpy as np
from morie.fn import splhf


@pytest.fixture()
def correlated_items():
    """6 items x 100 respondents from a single latent factor."""
    rng = np.random.default_rng(42)
    latent = rng.standard_normal(100)
    return np.column_stack([
        latent + rng.standard_normal(100) * 0.3
        for _ in range(6)
    ])


class TestSplhf:
    """Tests for split-half reliability."""

    def test_correlated_positive(self, correlated_items):
        """Correlated items should yield positive split-half r."""
        result = splhf(correlated_items)
        assert result > 0

    def test_first_last_method(self, correlated_items):
        """Default 'first_last' method should work."""
        result = splhf(correlated_items, method="first_last")
        assert -1 <= result <= 1

    def test_odd_even_method(self, correlated_items):
        """'odd_even' method should work and give a positive value."""
        result = splhf(correlated_items, method="odd_even")
        assert result > 0

    def test_both_methods_finite(self, correlated_items):
        """Both methods should return finite values."""
        r1 = splhf(correlated_items, method="first_last")
        r2 = splhf(correlated_items, method="odd_even")
        assert np.isfinite(r1)
        assert np.isfinite(r2)

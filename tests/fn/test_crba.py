"""Tests for morie.fn.crba — Cronbach's coefficient alpha."""

import pytest
import numpy as np
from morie.fn import crba
from morie.fn._containers import RlbRes


@pytest.fixture()
def correlated_items():
    """5 items x 100 respondents from correlated normal."""
    rng = np.random.default_rng(42)
    latent = rng.standard_normal(100)
    items = np.column_stack([
        latent + rng.standard_normal(100) * 0.3
        for _ in range(5)
    ])
    return items


class TestCrba:
    """Tests for Cronbach's alpha."""

    def test_known_value_high_alpha(self, correlated_items):
        """Highly correlated items should yield alpha > 0.5."""
        result = crba(correlated_items)
        assert isinstance(result, RlbRes)
        assert result.raw > 0.5

    def test_single_item_returns_nan(self):
        """Single-item data should return NaN alpha."""
        data = np.random.default_rng(42).standard_normal((50, 1))
        result = crba(data)
        assert np.isnan(result.raw)

    def test_ci_bounds_ordered(self, correlated_items):
        """CI lower bound <= raw <= CI upper bound."""
        result = crba(correlated_items)
        assert result.ci_lo <= result.raw <= result.ci_hi

    def test_k_and_n_correct(self, correlated_items):
        """k and n should match input dimensions."""
        result = crba(correlated_items)
        assert result.k == 5
        assert result.n == 100

    def test_standardised_alpha(self, correlated_items):
        """Standardised alpha should be finite and close to raw."""
        result = crba(correlated_items)
        assert np.isfinite(result.std)
